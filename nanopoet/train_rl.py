"""
RL 训练 - 强化学习后训练

使用 REINFORCE 算法优化诗词生成模型，基于三个奖励目标：
1. 押韵检查（30%）
2. 格式检查（30%）
3. 作者风格匹配（40%）

关键特性：
1. 策略梯度算法（REINFORCE）
2. 使用 baseline 降低方差
3. 只对生成的内容计算梯度（不更新提示部分）
4. 支持 checkpoint 恢复训练
5. 学习率线性衰减
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanopoet.common import (
    CharTokenizer, encode_poem_prompt, decode_poem_str,
    CONTENT_START, CONTENT_END, filter_by_author, update_poem_author,
    AUTHOR_S, STYLE_T, STYLE_S
)
from nanopoet.dataset import load_raw_data, get_base_dir
from nanopoet.model import GPTLanguageModel
from nanopoet.reward import compute_reward, extract_format
from nanopoet.train_reward import BinaryClassifier


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的 checkpoint 文件"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoint_files = sorted(checkpoint_path.glob("step_*.pt"))
    if not checkpoint_files:
        return None

    return checkpoint_files[-1]


def generate_rl_prompt(train_authors, train_styles):
    """
    生成 RL 训练的提示

    随机选择作者和风格，50% 概率包含每个元数据
    模仿 SFT 训练策略，增加训练多样性

    Args:
        train_authors: 作者列表
        train_styles: 风格列表

    Returns:
        dict: {'author': str, 'style': str} 或其子集
    """
    prompt_dict = {}

    # 50% 概率包含作者
    if random.random() > 0.5:
        prompt_dict['author'] = random.choice(train_authors)

    # 50% 概率包含风格
    if random.random() > 0.5:
        prompt_dict['style'] = random.choice(train_styles)

    return prompt_dict


@torch.no_grad()
def sample_from_model(model, tokenizer, prompt_str, num_samples, max_new_tokens,
                     block_size, device, temperature=1.0, top_k=50):
    """
    从模型采样生成多个样本

    Args:
        model: GPT 模型
        tokenizer: 分词器
        prompt_str: 提示字符串
        num_samples: 采样数量
        max_new_tokens: 最大生成 token 数
        block_size: 模型的上下文窗口大小
        device: 设备
        temperature: 采样温度
        top_k: Top-K 采样

    Returns:
        List[str]: 生成的样本列表
    """
    model.eval()

    # 编码提示
    prompt_tokens = tokenizer.encode(prompt_str)
    end_token_id = tokenizer.encode(CONTENT_END)[0]

    # 生成样本
    generated_samples = []

    for _ in range(num_samples):
        # 准备输入
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # 生成序列
        for _ in range(max_new_tokens):
            # 截断到 block_size
            input_tokens = tokens[:, -block_size:] if tokens.size(1) > block_size else tokens

            # 前向传播
            logits, _ = model(input_tokens)
            logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            tokens = torch.cat([tokens, next_token], dim=1)

            # 检查是否生成了结束 token
            if next_token.item() == end_token_id:
                break

        # 解码
        sample_text = tokenizer.decode(tokens[0].tolist())
        generated_samples.append(sample_text)

    return generated_samples


def compute_sample_reward(sample_str, classifier, tokenizer, format_data, device):
    """
    计算单个样本的奖励

    Args:
        sample_str: 生成的样本字符串（完整的编码格式）
        classifier: 奖励分类器
        tokenizer: 分词器
        format_data: 格式数据
        device: 设备

    Returns:
        float: 总奖励值 (0-1)
    """
    try:
        # 使用 compute_reward 计算奖励
        reward_result = compute_reward(
            poem_str=sample_str,
            classifier=classifier,
            tokenizer=tokenizer,
            format_data=format_data,
            device=device
        )
        return reward_result['total_reward']
    except Exception as e:
        # 如果计算失败，返回最低奖励
        # print(f"计算奖励时出错: {e}")
        return 0.0


def train_rl(
    model: GPTLanguageModel,
    classifier,
    tokenizer,
    format_data: dict,
    train_authors: list,
    train_styles: list,
    device,
    num_samples_per_prompt=8,
    num_steps=1000,
    learning_rate=1e-5,
    temperature=1.0,
    top_k=50,
    max_new_tokens=100,
    eval_interval=10,
    save_interval=100,
    checkpoint_dir="./output/checkpoints/rl",
    output_path="./output/rl_model.pt",
):
    """
    RL 训练主函数

    Args:
        model: GPT 模型（SFT 训练的结果）
        classifier: 奖励分类器
        tokenizer: 分词器
        format_data: 格式数据字典
        train_authors: 训练作者列表
        train_styles: 训练风格列表
        device: 设备
        num_samples_per_prompt: 每个提示采样的样本数
        num_steps: 总训练步数
        learning_rate: 学习率
        temperature: 采样温度
        top_k: Top-K 采样
        max_new_tokens: 最大生成 token 数
        eval_interval: 评估间隔
        save_interval: 保存间隔
        checkpoint_dir: checkpoint 保存目录
        output_path: 最终模型保存路径
    """
    # 创建必要的目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 尝试从最新的 checkpoint 恢复训练
    start_step = 0
    reward_history = []
    latest_checkpoint_file = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_file:
        print(f"\n发现 checkpoint，从 {latest_checkpoint_file} 恢复训练...")
        checkpoint = torch.load(latest_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        reward_history = checkpoint.get('reward_history', [])
        print(f"从步数 {start_step} 继续训练")
    else:
        print("\n未发现 checkpoint，从头开始训练...")

    block_size = model.block_size

    print(f"\n训练配置:")
    print(f"  总步数: {num_steps}")
    print(f"  每步采样数: {num_samples_per_prompt}")
    print(f"  学习率: {learning_rate:.2e}")
    print(f"  采样温度: {temperature}")
    print(f"  Top-K: {top_k}")
    print(f"  最大生成 tokens: {max_new_tokens}")
    print(f"  评估间隔: {eval_interval}")
    print(f"  保存间隔: {save_interval}")
    print()

    # 训练循环
    print("=" * 70)
    print("开始 RL 训练")
    print("=" * 70)

    for step in range(start_step, num_steps):
        # ========== 步骤 1: 生成提示 ==========
        prompt_dict = generate_rl_prompt(train_authors, train_styles)
        prompt_str = encode_poem_prompt(**prompt_dict) + CONTENT_START

        # ========== 步骤 2: 采样生成样本 ==========
        samples = sample_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt_str=prompt_str,
            num_samples=num_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            device=device,
            temperature=temperature,
            top_k=top_k
        )

        # ========== 步骤 3: 计算奖励 ==========
        rewards = []
        for sample in samples:
            reward = compute_sample_reward(sample, classifier, tokenizer, format_data, device)
            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_reward = rewards_tensor.mean().item()

        # ========== 步骤 4: 计算优势（Advantage） ==========
        # 使用当前批次的均值作为 baseline
        advantages = rewards_tensor - rewards_tensor.mean()

        # ========== 步骤 5: 策略梯度更新 ==========
        model.train()
        optimizer.zero_grad()

        # 对每个样本计算策略梯度
        total_loss = 0.0

        for sample, advantage in zip(samples, advantages):
            # 编码样本
            sample_tokens = tokenizer.encode(sample)

            # 截断到 block_size + 1（输入和目标）
            if len(sample_tokens) > block_size + 1:
                sample_tokens = sample_tokens[:block_size + 1]

            if len(sample_tokens) < 2:
                continue  # 样本太短，跳过

            # 准备输入和目标
            inputs = torch.tensor([sample_tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([sample_tokens[1:]], dtype=torch.long, device=device)

            # 计算提示长度（不对提示部分计算梯度）
            prefix_length = len(tokenizer.encode(prompt_str))

            # 创建 mask：只对生成的部分计算 loss
            mask = torch.zeros_like(targets[0], dtype=torch.float32)
            if prefix_length < len(targets[0]):
                mask[prefix_length:] = 1

            if mask.sum() == 0:
                continue  # 没有生成内容，跳过

            # 前向传播
            logits, _ = model(inputs)

            # 计算每个 token 的对数概率
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_log_probs = log_probs[range(len(targets[0])), targets[0]]

            # 只对生成的部分应用 advantage
            masked_log_probs = token_log_probs * mask

            # 策略梯度目标：log_prob * advantage
            # Loss = -Σ log_prob * advantage（最大化目标 = 最小化负目标）
            pg_loss = -(masked_log_probs * advantage).sum() / mask.sum()

            total_loss += pg_loss

        # 归一化 loss
        total_loss = total_loss / num_samples_per_prompt

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 学习率衰减（线性）
        lr_mult = 1.0 - step / num_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * lr_mult

        # 更新参数
        optimizer.step()

        # ========== 记录和输出 ==========
        reward_history.append(mean_reward)

        # 定期输出
        if step % eval_interval == 0 or step == num_steps - 1:
            # 计算移动平均
            window_size = min(50, len(reward_history))
            moving_avg = sum(reward_history[-window_size:]) / window_size

            # 计算趋势
            if len(reward_history) >= 20:
                recent_10 = sum(reward_history[-10:]) / 10
                previous_10 = sum(reward_history[-20:-10]) / 10
                trend = recent_10 - previous_10
                trend_symbol = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
            else:
                trend_symbol = "→"

            print(f"步数 {step:4d}/{num_steps} | "
                  f"奖励: {mean_reward:.4f} | "
                  f"移动平均: {moving_avg:.4f} {trend_symbol} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"LR: {lr_mult:.4f}")

            # 展示最佳样本
            if step % (eval_interval * 5) == 0 and step > 0:
                best_idx = rewards.index(max(rewards))
                best_sample = samples[best_idx]
                best_reward = rewards[best_idx]

                try:
                    poem_dict = decode_poem_str(best_sample)
                    print(f"  最佳样本 (奖励={best_reward:.4f}):")
                    print(f"    作者: {poem_dict.get('author', '(无)')}")
                    print(f"    风格: {poem_dict.get('style', '(无)')}")
                    print(f"    内容: {poem_dict.get('content', '(无内容)')[:40]}...")
                except Exception:
                    pass

        # ========== 保存 checkpoint ==========
        if step > 0 and (step % save_interval == 0 or step == num_steps - 1):
            checkpoint_file = checkpoint_path / f"step_{step:05d}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_reward': mean_reward,
                'reward_history': reward_history,
            }, checkpoint_file)
            print(f"  ✓ Checkpoint 已保存: {checkpoint_file}")

    # ========== 保存最终模型 ==========
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_reward': reward_history[-1] if reward_history else 0.0,
        'reward_history': reward_history,
    }, output_path)

    print("\n" + "=" * 70)
    print("RL 训练完成！")
    print("=" * 70)
    print(f"最终模型已保存到: {output_path}")

    if reward_history:
        print(f"\n训练总结:")
        print(f"  初始平均奖励: {reward_history[0]:.4f}")
        print(f"  最终平均奖励: {reward_history[-1]:.4f}")
        print(f"  总体改进: {reward_history[-1] - reward_history[0]:+.4f}")

    return model


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("RL 训练（强化学习后训练）")
    print("=" * 70)

    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    data = load_raw_data("../raw")

    # 初始化分词器
    tokenizer = CharTokenizer("".join(["".join(list(d.values())) for d in data]))
    print(f"词表大小: {tokenizer.vocab_size}")

    # 准备格式数据和作者/风格列表
    print("\n准备训练数据...")
    filtered_data = [update_poem_author(d) for d in data if filter_by_author(d)]
    format_data = extract_format(filtered_data)
    train_authors = AUTHOR_S
    train_styles = STYLE_T + STYLE_S
    print(f"  格式数量: {len(format_data)}")
    print(f"  作者数量: {len(train_authors)}")
    print(f"  风格数量: {len(train_styles)}")

    # 获取基础目录
    base_dir = get_base_dir()

    # 加载 SFT 模型
    sft_model_path = f"{base_dir}/sft_model.pt"
    print(f"\n加载 SFT 模型: {sft_model_path}")

    if not Path(sft_model_path).exists():
        print(f"错误：找不到 SFT 模型 {sft_model_path}")
        print("请先运行 SFT 训练")
        return

    # 初始化模型
    block_size = 256
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_size=256,
        block_size=block_size,
        layer_num=8,
        head_num=8,
        dropout=0.0,  # RL 训练不需要 dropout
    ).to(device)

    # 加载 SFT 权重
    sft_state = torch.load(sft_model_path, map_location=device)
    model.load_state_dict(sft_state, strict=True)
    print("✓ SFT 模型加载成功")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 加载奖励分类器
    classifier_path = f"{base_dir}/reward_classifier.pt"
    print(f"\n加载奖励分类器: {classifier_path}")

    if not Path(classifier_path).exists():
        print(f"错误：找不到奖励分类器 {classifier_path}")
        print("请先运行奖励模型训练")
        return

    # 创建分类器
    reward_gpt = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_size=256,
        block_size=block_size,
        layer_num=8,
        head_num=8,
        dropout=0.0,
    ).to(device)

    classifier = BinaryClassifier(
        gpt_model=reward_gpt,
        freeze_base=True,
        num_bidirectional_layers=2
    ).to(device)

    # 加载分类器权重
    classifier_checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(classifier_checkpoint['classifier_state_dict'])
    classifier.eval()
    print("✓ 奖励分类器加载成功")

    # 开始 RL 训练
    train_rl(
        model=model,
        classifier=classifier,
        tokenizer=tokenizer,
        format_data=format_data,
        train_authors=train_authors,
        train_styles=train_styles,
        device=device,
        num_samples_per_prompt=8,
        num_steps=1000,
        learning_rate=1e-5,
        temperature=1.0,
        top_k=50,
        max_new_tokens=100,
        eval_interval=10,
        save_interval=100,
        checkpoint_dir=f"{base_dir}/checkpoints/rl",
        output_path=f"{base_dir}/rl_model.pt",
    )


if __name__ == "__main__":
    main()
