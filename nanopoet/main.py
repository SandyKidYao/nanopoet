import torch
from pathlib import Path

from nanopoet.app import run_app
from nanopoet.common import CharTokenizer, filter_poem, update_poem_author, AUTHOR_S, STYLE_T, STYLE_S, filter_by_author
from nanopoet.dataset import load_raw_data, split_data, get_base_dir
from nanopoet.model import GPTLanguageModel
from nanopoet.train_mid import train_mid
from nanopoet.train_pre import train_pre
from nanopoet.train_sft import train_sft
from nanopoet.train_reward import train_reward_classifier, BinaryClassifier
from nanopoet.train_rl import train_rl
from nanopoet.reward import extract_format

def pre(model, tokenizer, device, train, val, base_dir):
    print("\n" + "=" * 70)
    print("开始 Pre Train 预训练...")
    print("=" * 70)
    batch_size = 64
    train_pre(
        model=model,
        tokenizer=tokenizer,
        device=device,
        train_poems=train,
        val_poems=val,
        batch_size=batch_size,
        learning_rate=3e-4,
        grad_clip=1.0,
        total_epochs=10,
        eval_interval=len(train) // batch_size // 2,
        eval_iters=len(val) // batch_size,
        checkpoint_dir=f"{base_dir}/checkpoints/pre",
        output_path=f"{base_dir}/pre_train_model.pt",
    )

def mid(model, tokenizer, device, train, val, base_dir):
    print("\n" + "=" * 70)
    print("开始 Mid Train 中期训练...")
    print("=" * 70)
    batch_size = 32
    train_mid(
        model=model,
        tokenizer=tokenizer,
        train_poems=train,
        val_poems=val,
        device=device,
        batch_size=batch_size,
        learning_rate=3e-4,
        init_lr_frac=0.5,
        warmdown_start_ratio=0.8,
        final_lr_frac=0.0,
        grad_clip=1.0,
        total_epochs=10,
        eval_interval=len(train) // batch_size // 2,
        eval_iters=len(val) // batch_size,
        checkpoint_dir=f"{base_dir}/checkpoints/mid",
        output_path=f"{base_dir}/mid_train_model.pt",
        pretrain_model_path=f"{base_dir}/pre_train_model.pt",
    )

def sft(model, tokenizer, device, train, val, base_dir):
    print("\n" + "=" * 70)
    print("开始 SFT 监督微调...")
    print("=" * 70)
    batch_size = 16
    filtered_train_poems = [update_poem_author(p) for p in train if filter_poem(p)]
    filtered_val_poems = [update_poem_author(p) for p in val if filter_poem(p)]
    print(f"过滤后训练集大小: {len(filtered_train_poems)}")
    print(f"过滤后验证集大小: {len(filtered_val_poems)}")

    train_sft(
        model=model,
        tokenizer=tokenizer,
        train_poems=filtered_train_poems,
        val_poems=filtered_val_poems,
        device=device,
        batch_size=batch_size,
        learning_rate=3e-4,
        init_lr_frac=0.2,
        grad_clip=1.0,
        total_epochs=10,
        eval_interval=len(filtered_train_poems) // batch_size // 2,
        eval_iters=len(filtered_val_poems) // batch_size,
        checkpoint_dir=f"{base_dir}/checkpoints/sft",
        output_path=f"{base_dir}/sft_model.pt",
        pretrain_model_path=f"{base_dir}/mid_train_model.pt",
    )

def reward(model, tokenizer, device, train, val, base_dir):
    """
    训练 Reward 分类器（作者风格匹配）

    用于强化学习阶段的奖励信号，判断"作者-诗词"配对是否合理。

    关键特性：
    - 基于 SFT 模型，添加双向 Transformer 层
    - 冻结 GPT 权重，只训练双向层和分类头（~1.6M 参数）
    - 使用全局平均池化而非只取最后一个 token
    - 充分利用数据：所有符合条件的诗词都作为正样本，并生成等量负样本
    """
    print("\n" + "=" * 70)
    print("开始 Reward 分类器训练...")
    print("=" * 70)

    # 使用过滤后的数据（与 SFT 相同的作者和风格范围）
    filtered_train_poems = [update_poem_author(p) for p in train if filter_by_author(p)]
    filtered_val_poems = [update_poem_author(p) for p in val if filter_by_author(p)]
    print(f"过滤后训练集大小: {len(filtered_train_poems)}")
    print(f"过滤后验证集大小: {len(filtered_val_poems)}")

    # 训练配置
    batch_size = 16

    # 训练作者风格匹配分类器
    train_reward_classifier(
        gpt_model=model,
        tokenizer=tokenizer,
        train_poems=filtered_train_poems,
        val_poems=filtered_val_poems,
        device=device,
        batch_size=batch_size,
        learning_rate=1e-4,  # 较小的学习率
        total_epochs=10,
        eval_interval=100,  # 每100步评估一次
        eval_iters=10,
        checkpoint_dir=f"{base_dir}/checkpoints/reward",
        output_path=f"{base_dir}/reward_classifier.pt",
        pretrain_model_path=f"{base_dir}/sft_model.pt",  # 基于 SFT 模型
        num_bidirectional_layers=2,
    )

def rl(model, tokenizer, device, train, val, base_dir):
    """
    RL 训练（强化学习后训练）

    使用 REINFORCE 算法优化模型，基于三个奖励目标。

    关键特性：
    - 基于 SFT 模型进行强化学习训练
    - 使用策略梯度算法（REINFORCE）
    - 三个奖励目标：押韵（30%）、格式（30%）、作者风格（40%）
    - 学习率线性衰减，支持 checkpoint 恢复
    """
    print("\n" + "=" * 70)
    print("开始 RL 训练（强化学习）...")
    print("=" * 70)

    # 准备训练数据
    filtered_data = [update_poem_author(p) for p in train + val if filter_by_author(p)]
    format_data = extract_format(filtered_data)
    train_authors = AUTHOR_S
    train_styles = STYLE_T + STYLE_S

    print(f"训练数据统计:")
    print(f"  格式数量: {len(format_data)}")
    print(f"  作者数量: {len(train_authors)}")
    print(f"  风格数量: {len(train_styles)}")

    # 加载 SFT 模型
    sft_model_path = f"{base_dir}/sft_model.pt"
    print(f"\n加载 SFT 模型: {sft_model_path}")
    sft_state = torch.load(sft_model_path, map_location=device)
    model.load_state_dict(sft_state, strict=True)
    model.dropout = 0.0  # RL 训练不需要 dropout
    print("✓ SFT 模型加载成功")

    # 加载奖励分类器
    classifier_path = f"{base_dir}/reward_classifier.pt"
    print(f"\n加载奖励分类器: {classifier_path}")

    reward_gpt = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_size=model.emb_size,
        block_size=model.block_size,
        layer_num=model.layer_num,
        head_num=model.head_num,
        dropout=0.0,
    ).to(device)

    classifier = BinaryClassifier(
        gpt_model=reward_gpt,
        freeze_base=True,
        num_bidirectional_layers=2
    ).to(device)

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
        num_samples_per_prompt=8,    # 每个提示采样 8 个样本
        num_steps=1000,               # 训练 1000 步
        learning_rate=1e-5,           # 较小的学习率
        temperature=1.0,              # 采样温度
        top_k=50,                     # Top-K 采样
        max_new_tokens=100,           # 最大生成 token 数
        eval_interval=10,             # 每 10 步输出一次
        save_interval=100,            # 每 100 步保存一次
        checkpoint_dir=f"{base_dir}/checkpoints/rl",
        output_path=f"{base_dir}/rl_model.pt",
    )

def start_app(model, tokenizer, device, base_dir):
    print("\n" + "=" * 70)
    print("准备启动 Web 应用...")
    print("=" * 70)

    # 检查哪些模型文件存在，并加载它们
    potential_models = [
        {"name": "RL 模型", "path": f"{base_dir}/rl_model.pt"},
        {"name": "SFT 模型", "path": f"{base_dir}/sft_model.pt"},
        {"name": "Mid Train 模型", "path": f"{base_dir}/mid_train_model.pt"},
        {"name": "Pre Train 模型", "path": f"{base_dir}/pre_train_model.pt"},
    ]

    loaded_models = []
    for model_info in potential_models:
        model_path = Path(model_info['path'])
        if model_path.exists():
            print(f"\n加载模型: {model_info['name']}")
            print(f"  路径: {model_path}")

            # 加载模型
            checkpoint = torch.load(model_path, map_location=device)

            # 处理不同的保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # RL 模型格式
                state_dict = checkpoint['model_state_dict']
            else:
                # 其他模型格式（直接保存的 state_dict）
                state_dict = checkpoint

            loaded_model = GPTLanguageModel(
                vocab_size=tokenizer.vocab_size,
                emb_size=model.emb_size,
                block_size=model.block_size,
                layer_num=model.layer_num,
                head_num=model.head_num,
                dropout=0.0,  # 推理时不需要 dropout
            )
            loaded_model.load_state_dict(state_dict)
            loaded_model = loaded_model.to(device)
            loaded_model.eval()

            loaded_models.append({
                "name": model_info['name'],
                "model": loaded_model
            })
            print(f"  ✓ 加载成功")

    if not loaded_models:
        print("\n❌ 错误：未找到任何训练好的模型文件！")
        print("请先运行训练代码生成模型文件。")
        print("\n可用的训练阶段：")
        print("  1. Pre Train  - 预训练阶段")
        print("  2. Mid Train  - 中期训练阶段")
        print("  3. SFT        - 监督微调阶段")
        print("\n取消相应代码段的注释即可运行训练。")
        return

    # 提取作者和风格列表
    authors_list = AUTHOR_S
    styles_list = STYLE_T + STYLE_S

    # 启动 Web 应用
    run_app(
        models=loaded_models,
        tokenizer=tokenizer,
        device=device,
        authors_list=authors_list,
        styles_list=styles_list,
        port=54321
    )

def main():
    # ========== 初始化设备和数据 ==========
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    data = load_raw_data("../raw")
    train, val = split_data(data)
    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(val)}")

    # 初始化分词器
    tokenizer = CharTokenizer("".join(["".join(list(d.values())) for d in data]))
    print(f"词表大小: {tokenizer.vocab_size}")

    # 初始化模型
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_size=256,
        block_size=256,
        layer_num=8,
        head_num=8,
        dropout=0.1,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    base_dir = get_base_dir()

    # ========== Pre Train 预训练阶段 ==========
    # 取消注释以运行预训练
    # pre(model, tokenizer, device, train, val, base_dir)

    # ========== Mid Train 中期训练阶段 ==========
    # 取消注释以运行中期训练
    # mid(model, tokenizer, device, train, val, base_dir)

    # ========== SFT 监督微调阶段 ==========
    # 取消注释以运行 SFT 训练
    # sft(model, tokenizer, device, train, val, base_dir)

    # ========== Reward 分类器训练阶段 ==========
    # 取消注释以运行 Reward 分类器训练
    # reward(model, tokenizer, device, train, val, base_dir)

    # ========== RL 强化学习训练阶段 ==========
    # 取消注释以运行 RL 训练
    # rl(model, tokenizer, device, train, val, base_dir)

    # ========== 启动 Web 应用 ==========
    start_app(model, tokenizer, device, base_dir)


if __name__ == '__main__':
    main()
