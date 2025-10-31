import torch
from pathlib import Path

from nanopoet.app import run_app
from nanopoet.common import CharTokenizer, filter_poem, update_poem_author, AUTHOR_S, STYLE_T, STYLE_S
from nanopoet.dataset import load_raw_data, split_data, get_base_dir
from nanopoet.model import GPTLanguageModel
from nanopoet.train_mid import train_mid
from nanopoet.train_pre import train_pre
from nanopoet.train_sft import train_sft

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

def start_app(model, tokenizer, device, base_dir):
    print("\n" + "=" * 70)
    print("准备启动 Web 应用...")
    print("=" * 70)

    # 检查哪些模型文件存在，并加载它们
    potential_models = [
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
            state_dict = torch.load(model_path, map_location=device)
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
    sft(model, tokenizer, device, train, val, base_dir)

    # ========== 启动 Web 应用 ==========
    # start_app(model, tokenizer, device, base_dir)


if __name__ == '__main__':
    main()
