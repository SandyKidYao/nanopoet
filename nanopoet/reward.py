"""
奖励计算模块

提供三种奖励目标的计算函数：
1. 押韵检查（基于规则）
2. 格式检查（基于规则）
3. 作者风格匹配（基于神经网络分类器）

用于强化学习阶段的奖励信号计算。
"""

import torch
from pypinyin import pinyin, Style
from collections import Counter

from nanopoet.common import decode_poem_str


# ==================== 奖励目标 1: 押韵检查 ====================

def check_rhyme(content: str) -> float:
    """
    押韵奖励函数

    要求：每个句号结尾的最后一个字都需要押韵

    Args:
        content: 诗词内容

    Returns:
        0-1 的奖励值，1 表示完美押韵
    """
    # 用句号将诗词切分，取每一句最后一个字符
    sentences = [s for s in content.split("。") if s.strip()]
    if len(sentences) < 2:
        # 句子太少，无法检查押韵
        return 0.0

    # 提取每句最后一个字的韵母
    end_chars = [s[-1] for s in sentences]
    end_rhymes = [pinyin(c, style=Style.FINALS)[0][0] for c in end_chars]

    # 统计最常见的韵母出现次数
    rhyme_counter = Counter(end_rhymes)
    most_common_rhyme = max(list(rhyme_counter.values()))

    # 如果有 70% 的句子都押同一个韵，则认为符合要求
    target_rhyme_number = int(len(end_chars) * 0.7)
    return min(float(most_common_rhyme) / float(target_rhyme_number), 1.0)


# ==================== 奖励目标 2: 格式检查 ====================

def generate_format_code(content: str) -> str:
    """
    对诗词进行编码，返回每一句的字数信息

    例如：七言律诗 -> "7-7-7-7-7-7-7-7"

    Args:
        content: 诗词内容

    Returns:
        格式编码字符串
    """
    tmp = content.split("。")
    lines = []
    for c in tmp:
        lines.extend(c.split("，"))
    lines = [l for l in lines if l.strip()]
    return "-".join([str(len(l)) for l in lines])


def extract_format(data: list[dict]) -> dict[str, list[str]]:
    """
    根据数据集统计各个形式诗词的格式规则

    Args:
        data: 诗词数据列表，每个元素为 {"style": ..., "content": ...}

    Returns:
        {style: [format1, format2, ...]} 的字典
    """
    style_formats = {}
    for poem in data:
        style = poem["style"]
        format_code = generate_format_code(poem["content"])
        if style not in style_formats:
            style_formats[style] = []
        if format_code not in style_formats[style]:
            style_formats[style].append(format_code)
    return style_formats


def check_format(style: str, content: str, format_data: dict[str, list[str]]) -> int:
    """
    格式检查函数（严格模式）

    Args:
        style: 诗词风格（如"七言律诗"）
        content: 诗词内容
        format_data: 格式数据字典（由 extract_format 生成）

    Returns:
        1 表示格式完全符合，0 表示不符合
    """
    style_formats = format_data.get(style)
    if style_formats is None:
        return 0

    format_code = generate_format_code(content)
    return 1 if format_code in style_formats else 0


# ==================== 综合评价函数 ====================

def compute_reward(
    poem_str: str,
    classifier,
    tokenizer,
    format_data: dict,
    device: str,
    weights: dict = None
) -> dict:
    """
    综合评价函数，整合三个奖励目标

    Args:
        poem_str: 编码后的诗词字符串（如 "BA李白aC床前明月光...c"）
        classifier: 作者风格匹配分类器（BinaryClassifier 实例）
        tokenizer: 分词器
        format_data: 格式数据字典
        device: 设备
        weights: 各项权重，默认 {'rhyme': 0.3, 'format': 0.3, 'author_style': 0.4}

    Returns:
        包含各项分数和总分的字典：
        {
            'rhyme_score': float,
            'format_score': float,
            'author_style_score': float,
            'total_reward': float,
            'details': {
                'rhyme': str,
                'format': str,
                'author_style': str
            }
        }
    """
    # 默认权重
    if weights is None:
        weights = {
            'rhyme': 0.3,         # 押韵权重
            'format': 0.3,        # 格式权重
            'author_style': 0.4   # 作者风格匹配权重
        }

    # 解码诗词字符串
    poem_dict = decode_poem_str(poem_str)

    # 初始化结果
    result = {
        'rhyme_score': 0.0,
        'format_score': 0.0,
        'author_style_score': 0.0,
        'total_reward': 0.0,
        'details': {}
    }

    # 1. 押韵检查
    if 'content' in poem_dict:
        rhyme_score = check_rhyme(poem_dict['content'])
        result['rhyme_score'] = rhyme_score
        result['details']['rhyme'] = f"{rhyme_score:.2%}"

    # 2. 格式检查
    if 'style' in poem_dict and 'content' in poem_dict:
        format_score = check_format(poem_dict['style'], poem_dict['content'], format_data)
        result['format_score'] = float(format_score)
        result['details']['format'] = "✓ 通过" if format_score == 1 else "✗ 不通过"
    else:
        # 如果没有 style 字段，格式检查得满分（不做限制）
        result['format_score'] = 1.0
        result['details']['format'] = "⊘ 无格式要求"

    # 3. 作者风格匹配
    if 'author' in poem_dict and 'content' in poem_dict:
        classifier.eval()
        test_ids = torch.tensor([tokenizer.encode(poem_str)], device=device)
        with torch.no_grad():
            logits = classifier(test_ids)
            prob = torch.sigmoid(logits).item()
        result['author_style_score'] = prob
        result['details']['author_style'] = f"{prob:.2%}"
    else:
        # 如果没有 author 字段，作者匹配得满分（不做限制）
        result['author_style_score'] = 1.0
        result['details']['author_style'] = "⊘ 无作者要求"

    # 4. 计算总奖励（加权平均）
    total_reward = (
        weights['rhyme'] * result['rhyme_score'] +
        weights['format'] * result['format_score'] +
        weights['author_style'] * result['author_style_score']
    )
    result['total_reward'] = total_reward

    return result


def print_reward_analysis(poem_str: str, reward_result: dict):
    """
    打印奖励分析结果

    Args:
        poem_str: 编码后的诗词字符串
        reward_result: compute_reward 返回的结果字典
    """
    poem_dict = decode_poem_str(poem_str)

    print("=" * 70)
    print("诗词评价分析")
    print("=" * 70)

    # 诗词信息
    print("\n【诗词信息】")
    if 'author' in poem_dict:
        print(f"  作者: {poem_dict['author']}")
    if 'style' in poem_dict:
        print(f"  形式: {poem_dict['style']}")
    if 'title' in poem_dict:
        print(f"  标题: {poem_dict['title']}")
    if 'content' in poem_dict:
        print(f"  内容: {poem_dict['content']}")

    # 评分详情
    print("\n【评分详情】")
    print(f"  押韵检查:     {reward_result['details'].get('rhyme', 'N/A'):>10s} (权重: 30%)")
    print(f"  格式检查:     {reward_result['details'].get('format', 'N/A'):>10s} (权重: 30%)")
    print(f"  作者风格匹配: {reward_result['details'].get('author_style', 'N/A'):>10s} (权重: 40%)")

    # 总分
    print("\n【综合评分】")
    print(f"  总奖励: {reward_result['total_reward']:.2%}")

    # 评级
    score = reward_result['total_reward']
    if score >= 0.9:
        grade = "优秀 ★★★★★"
    elif score >= 0.8:
        grade = "良好 ★★★★"
    elif score >= 0.7:
        grade = "中等 ★★★"
    elif score >= 0.6:
        grade = "及格 ★★"
    else:
        grade = "待改进 ★"
    print(f"  评级: {grade}")
    print("=" * 70)
