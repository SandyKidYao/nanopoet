import random
import torch
import torch.nn.functional as F
from pathlib import Path

from nanopoet.common import (
    CONTENT_START,
    CONTENT_END,
    PADDING,
    encode_poem,
    encode_poem_prompt
)
from nanopoet.model import GPTLanguageModel, save_checkpoint, load_checkpoint


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件（通过文件名自然排序）"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # 查找所有 epoch_*.pt 文件
    checkpoint_files = sorted(checkpoint_path.glob("epoch_*.pt"))
    if not checkpoint_files:
        return None

    # 返回最新的（文件名自然有序，最后一个就是最新的）
    return checkpoint_files[-1]


def erase_metadata(poem):
    """
    随机擦除诗词的元数据

    让模型学会在各种情况下生成诗词（有完整元数据、部分元数据、无元数据等）
    """
    new_poem = {
        "content": poem["content"],
    }
    if random.random() < 0.5:
        new_poem["author"] = poem["author"]
    if random.random() < 0.5:
        new_poem["title"] = poem["title"]
    if random.random() < 0.5:
        new_poem["style"] = poem["style"]
    return new_poem


def get_batch(poems_encoded, batch_size, start_token_id, end_token_id, pad_token_id, block_size, device):
    """
    数据加载函数（SFT 专用）

    与 Mid Train 不同，SFT 只计算诗词内容部分的 loss，
    元数据和 padding 部分使用 -1 标记（ignore_index）
    """
    # 随机抽取一个批次的诗词
    indices = torch.randint(len(poems_encoded), (batch_size,))
    batch = [poems_encoded[i] for i in indices]

    # 截断过长的数据
    batch = [p[:block_size] if len(p) > block_size else p for p in batch]
    max_len = max(len(p) for p in batch)

    batch_x = []
    batch_y = []

    for poem in batch:
        if len(poem) > 1:
            x = poem[:-1]  # 输入序列
            y = poem[1:]   # 目标序列

            # 创建 mask：找到 START_TOKEN 的位置
            # 注意：在 y 中查找，因为 y 是我们要预测的目标
            y_list = y.tolist()

            # 初始化为 -1（全部不计算 loss）
            y_masked = torch.full_like(y, -1)

            # 找到 START_TOKEN 在 y 中的位置
            try:
                start_idx = y_list.index(start_token_id)
                # 找到 END_TOKEN 在 y 中的位置
                end_idx = y_list.index(end_token_id, start_idx)
                # 从 START_TOKEN 到 END_TOKEN（包括两端）设置为真实 token
                y_masked[start_idx:end_idx + 1] = y[start_idx:end_idx + 1]
            except ValueError:
                # 如果没找到 START_TOKEN 或 END_TOKEN，整个序列都不计算 loss
                pass

            # 填充
            pad_len = max_len - 1 - len(x)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                y_masked = torch.cat([y_masked, torch.full((pad_len,), -1, dtype=torch.long)])

            batch_x.append(x)
            batch_y.append(y_masked)

    return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


def estimate_loss(train_data, val_data, model, tokenizer, eval_iters, batch_size, start_token_id,
                  end_token_id, pad_token_id, block_size, device):
    """评估损失函数（SFT 版本）"""
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, start_token_id, end_token_id, pad_token_id, block_size, device)
            logits, _ = model(X, Y)
            losses[k] = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                Y.view(-1),
                ignore_index=-1
            ).item()
        out[split] = losses.mean()
    return out


def generate_samples(model, tokenizer, device, train_authors, train_styles, end_token_id, block_size):
    """生成多种测试样本"""
    # 从训练数据的作者和风格中随机取一个
    test_author = random.choice(train_authors)
    test_style = random.choice(train_styles)

    # 生成多种 prompt 来看看实际使用时不同情况的样本
    test_prompts = [
        ("无提示", encode_poem_prompt()),
        (f"作者: {test_author}", encode_poem_prompt(author=test_author)),
        (f"风格: {test_style}", encode_poem_prompt(style=test_style)),
        (f"作者+风格", encode_poem_prompt(author=test_author, style=test_style)),
        (f"完整信息", encode_poem_prompt(author=test_author, style=test_style, title="咏柳")),
    ]

    samples = []
    for label, prompt in test_prompts:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        # 生成时不再限制长度，而是让模型生成到结束符位置
        generated = model.generate(context, max_new_tokens=block_size, stop_token_ids=[end_token_id])
        sample = tokenizer.decode(generated[0].tolist())
        samples.append((label, sample))

    return samples


def train_sft(
        model: GPTLanguageModel,
        tokenizer,
        train_poems,
        val_poems,
        device,
        batch_size,
        learning_rate,
        grad_clip,
        total_epochs,
        eval_interval,
        eval_iters,
        checkpoint_dir,
        output_path,
        pretrain_model_path=None,
        init_lr_frac=0.02,
):
    """
    SFT (Supervised Fine-Tuning) 训练

    在 Mid Train 的基础上，进一步特化模型学习根据元数据生成诗词。
    与 Mid Train 的主要区别：
    1. 使用过滤后的数据（特定作者和风格）
    2. 随机擦除元数据，让模型适应各种输入情况
    3. 只计算诗词内容部分的 loss，忽略元数据和 padding
    4. 使用更小的学习率，从训练开始就线性衰减

    Args:
        model: GPT 模型
        tokenizer: 分词器
        train_poems: 训练集诗词数据（已过滤）
        val_poems: 验证集诗词数据（已过滤）
        device: 设备
        batch_size: 批次大小
        learning_rate: 基础学习率
        grad_clip: 梯度裁剪阈值
        total_epochs: 总训练轮数
        eval_interval: 评估间隔（每隔多少步评估一次）
        eval_iters: 评估时的迭代次数
        checkpoint_dir: checkpoint 保存目录
        output_path: 最终模型保存路径
        pretrain_model_path: 预训练模型路径（通常是 mid train 的结果）
        init_lr_frac: 初始学习率倍率（相对于 learning_rate，默认2%）
    """
    # 创建必要的目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型（通常是 mid train 的结果）
    if pretrain_model_path:
        print(f"加载预训练模型: {pretrain_model_path}")
        pretrain_state = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(pretrain_state, strict=True)
        print("预训练模型加载成功")

    # 准备训练数据：随机擦除元数据
    train_erased = [erase_metadata(p) for p in train_poems]
    val_erased = [erase_metadata(p) for p in val_poems]

    # 使用特殊 token 编码每首诗词
    train_ps = [encode_poem(poem) for poem in train_erased]
    val_ps = [encode_poem(poem) for poem in val_erased]

    # 编码成 token IDs
    train_data = [torch.tensor(tokenizer.encode(poem), dtype=torch.long) for poem in train_ps]
    val_data = [torch.tensor(tokenizer.encode(poem), dtype=torch.long) for poem in val_ps]

    # 提取训练数据里的 author 和 style，用于后续生成样本
    train_authors = list(set([p["author"] for p in train_poems if "author" in p]))
    train_styles = list(set([p["style"] for p in train_poems if "style" in p]))

    # 获取特殊 token IDs
    pad_token_id = tokenizer.encode(PADDING)[0]
    start_token_id = tokenizer.encode(CONTENT_START)[0]
    end_token_id = tokenizer.encode(CONTENT_END)[0]
    block_size = model.block_size

    # 初始化优化器（使用较低的学习率）
    initial_lr = learning_rate * init_lr_frac
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    # 尝试从最新的 checkpoint 恢复训练
    start_epoch = 0
    global_step = 0
    latest_checkpoint_file = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_file:
        print(f"发现checkpoint，从 {latest_checkpoint_file} 恢复训练...")
        loaded_epoch, loaded_step = load_checkpoint(latest_checkpoint_file, model, optimizer)
        start_epoch = loaded_epoch + 1
        global_step = loaded_step
        print(f"从第 {start_epoch} 个epoch继续训练 (global_step={global_step})")
    else:
        print("未发现checkpoint，从头开始训练...")

    # 将模型移到设备上
    model = model.to(device)
    model.train()

    # 计算总步数（用于学习率调度）
    steps_per_epoch = len(train_data) // batch_size
    total_steps = total_epochs * steps_per_epoch

    print(f"\n训练配置:")
    print(f"  - 数据集大小: 训练={len(train_data)}, 验证={len(val_data)}")
    print(f"  - 总轮数: {total_epochs}")
    print(f"  - 每轮步数: {steps_per_epoch}")
    print(f"  - 总步数: {total_steps}")
    print(f"  - 初始学习率: {initial_lr:.2e} (基础LR的{init_lr_frac:.1%})")
    print(f"  - 学习率策略: 线性衰减到0")
    print(f"  - 梯度裁剪: {grad_clip}")

    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        print(f"\n===== Epoch {epoch + 1}/{total_epochs} =====")

        for step in range(steps_per_epoch):
            global_step += 1

            # 获取批次数据
            xb, yb = get_batch(train_data, batch_size, start_token_id, end_token_id,
                              pad_token_id, block_size, device)

            # 前向传播
            logits, _ = model(xb)

            # 计算 loss（ignore_index=-1 会自动忽略条件和padding部分）
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                yb.view(-1),
                ignore_index=-1
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 学习率调度：从训练开始就线性衰减
            lr_mult = 1.0 - global_step / total_steps
            current_lr = initial_lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.step()

            # 定期评估和生成样本
            if global_step % eval_interval == 0 or global_step == 1:
                model.eval()
                out = estimate_loss(train_data, val_data, model, tokenizer, eval_iters,
                                  batch_size, start_token_id, end_token_id, pad_token_id,
                                  block_size, device)
                print(f"Step {global_step:5d} | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f} | LR: {current_lr:.2e} (x{lr_mult:.3f})")

                # 生成多种样本
                if train_authors and train_styles:
                    samples = generate_samples(model, tokenizer, device, train_authors,
                                             train_styles, end_token_id, block_size)
                    for label, sample in samples:
                        print(f"  [{label}] {sample}")

                model.train()

        # 每个 epoch 结束时评估并保存 checkpoint
        model.eval()
        out = estimate_loss(train_data, val_data, model, tokenizer, eval_iters,
                          batch_size, start_token_id, end_token_id, pad_token_id,
                          block_size, device)
        train_loss = out['train'].item()
        val_loss = out['val'].item()
        print(f"Epoch {epoch + 1} 结束 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        model.train()

        # 保存 checkpoint
        checkpoint_file = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(model, optimizer, epoch, global_step, train_loss, val_loss, checkpoint_file)

    # 训练完成，最终评估和生成样本
    model.eval()
    out = estimate_loss(train_data, val_data, model, tokenizer, eval_iters,
                      batch_size, start_token_id, end_token_id, pad_token_id,
                      block_size, device)
    print(f"\nFinal | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f}")

    # 生成最终样本
    if train_authors and train_styles:
        print("\n最终样本:")
        samples = generate_samples(model, tokenizer, device, train_authors,
                                 train_styles, end_token_id, block_size)
        for label, sample in samples:
            print(f"  [{label}] {sample}")

    # 保存最终模型
    torch.save(model.state_dict(), output_path)
    print(f"\n训练完成！最终模型已保存到 {output_path}")

    return model
