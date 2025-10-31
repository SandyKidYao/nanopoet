import torch
import torch.nn.functional as F
from pathlib import Path

from nanopoet.common import BEGIN, PADDING, encode_poem
from nanopoet.model import GPTLanguageModel, generate_sample, save_checkpoint, load_checkpoint


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


def get_batch(poems_encoded, batch_size, pad_token_id, block_size, device):
    """
    数据加载函数（Mid Train 专用）

    处理不同长度的诗词序列，通过 padding 对齐，并返回 mask
    """
    # 随机抽取一个批次的诗词
    indices = torch.randint(len(poems_encoded), (batch_size,))
    batch = [poems_encoded[i] for i in indices]

    # 截断超长序列，补齐过短序列
    batch = [p[:block_size] if len(p) > block_size else p for p in batch]

    # 获取批次中的最大长度（优化：不强制补齐到 block_size）
    max_len = max(len(p) for p in batch)

    # 创建填充后的批次
    batch_x = []
    batch_y = []
    batch_mask = []

    for poem in batch:
        # 输入: poem[:-1], 目标: poem[1:]
        if len(poem) > 1:
            x = poem[:-1]
            y = poem[1:]

            # 计算需要填充的长度
            pad_len = max_len - 1 - len(x)

            if pad_len > 0:
                # 填充
                x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                y = torch.cat([y, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                # mask: 真实token为1，填充为0
                mask = torch.cat([
                    torch.ones(len(poem) - 1, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            else:
                # 不需要填充
                mask = torch.ones(len(x), dtype=torch.long)

            batch_x.append(x)
            batch_y.append(y)
            batch_mask.append(mask)

    return (
        torch.stack(batch_x).to(device),
        torch.stack(batch_y).to(device),
        torch.stack(batch_mask).to(device)
    )


def get_loss_with_mask(logits, targets, mask):
    """
    计算带 mask 的损失

    只计算非填充位置的 loss
    """
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    targets_flat = targets.view(B * T)
    mask_flat = mask.view(B * T)

    # 只计算 mask=1 的位置
    if mask_flat.sum() > 0:
        loss = F.cross_entropy(
            logits_flat[mask_flat == 1],
            targets_flat[mask_flat == 1]
        )
    else:
        loss = torch.tensor(0.0, device=logits.device)

    return loss


def estimate_loss(train_data, val_data, model, eval_iters, batch_size, pad_token_id, block_size, device):
    """评估损失函数（带 mask 的版本）"""
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, mask = get_batch(data, batch_size, pad_token_id, block_size, device)
            logits, _ = model(x, y)
            losses[k] = get_loss_with_mask(logits, y, mask).item()
        out[split] = losses.mean()
    return out


def train_mid(
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
        init_lr_frac=0.5,
        warmdown_start_ratio=0.8,
        final_lr_frac=0.0,
):
    """
    Mid Train 中期训练

    Args:
        model: GPT 模型
        tokenizer: 分词器
        train_poems: 训练集诗词数据
        val_poems: 验证集诗词数据
        device: 设备
        batch_size: 批次大小
        learning_rate: 学习率
        grad_clip: 梯度裁剪阈值
        total_epochs: 总训练轮数
        eval_interval: 评估间隔（每隔多少步评估一次）
        eval_iters: 评估时的迭代次数
        checkpoint_dir: checkpoint 保存目录
        output_path: 最终模型保存路径
        pretrain_model_path: 预训练模型路径（可选）
        init_lr_frac: 初始学习率倍率（相对于 learning_rate）
        warmdown_start_ratio: 学习率开始衰减的位置（相对于总步数）
        final_lr_frac: 最终学习率倍率
    """
    # 创建必要的目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型（如果提供）
    if pretrain_model_path:
        print(f"加载预训练模型: {pretrain_model_path}")
        pretrain_state = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(pretrain_state, strict=True)
        print("预训练模型加载成功")

    # 准备训练数据：使用特殊 token 编码每首诗词
    train_ps = [encode_poem(poem) for poem in train_poems]
    val_ps = [encode_poem(poem) for poem in val_poems]

    # 编码成 token IDs
    train_data = [torch.tensor(tokenizer.encode(poem), dtype=torch.long) for poem in train_ps]
    val_data = [torch.tensor(tokenizer.encode(poem), dtype=torch.long) for poem in val_ps]

    # 获取 padding token ID
    pad_token_id = tokenizer.encode(PADDING)[0]
    block_size = model.block_size

    # 初始化优化器（从较低的学习率开始）
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
    warmdown_start_step = int(warmdown_start_ratio * total_steps)

    print(f"\n训练配置:")
    print(f"  - 总轮数: {total_epochs}")
    print(f"  - 每轮步数: {steps_per_epoch}")
    print(f"  - 总步数: {total_steps}")
    print(f"  - 初始学习率: {initial_lr:.2e} (基础LR的{init_lr_frac:.1%})")
    print(f"  - 学习率衰减起点: step {warmdown_start_step} ({warmdown_start_ratio:.1%})")
    print(f"  - 梯度裁剪: {grad_clip}")

    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        print(f"\n===== Epoch {epoch + 1}/{total_epochs} =====")

        for step in range(steps_per_epoch):
            global_step += 1

            # 获取批次数据
            xb, yb, mask = get_batch(train_data, batch_size, pad_token_id, block_size, device)

            # 前向传播（忽略模型返回的 loss）
            logits, _ = model(xb, yb)

            # 计算带 mask 的 loss
            loss = get_loss_with_mask(logits, yb, mask)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 学习率调度
            if global_step < warmdown_start_step:
                # 前 80% 保持恒定
                lr_mult = 1.0
            else:
                # 后 20% 线性衰减
                progress = (total_steps - global_step) / (total_steps - warmdown_start_step)
                lr_mult = progress * 1.0 + (1 - progress) * final_lr_frac

            # 应用新的学习率
            current_lr = initial_lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.step()

            # 定期评估和生成样本
            if global_step % eval_interval == 0 or global_step == 1:
                model.eval()
                out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, pad_token_id, block_size, device)
                print(f"Step {global_step:5d} | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f} | LR: {current_lr:.2e} (x{lr_mult:.3f})")
                print(f"Sample: {generate_sample(model, tokenizer, device, BEGIN)}")
                model.train()

        # 每个 epoch 结束时评估并保存 checkpoint
        model.eval()
        out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, pad_token_id, block_size, device)
        train_loss = out['train'].item()
        val_loss = out['val'].item()
        print(f"Epoch {epoch + 1} 结束 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        model.train()

        # 保存 checkpoint
        checkpoint_file = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(model, optimizer, epoch, global_step, train_loss, val_loss, checkpoint_file)

    # 训练完成，最终评估和生成样本
    model.eval()
    out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, pad_token_id, block_size, device)
    print(f"\nFinal | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f}")
    print(f"Sample: {generate_sample(model, tokenizer, device, BEGIN)}")

    # 保存最终模型
    torch.save(model.state_dict(), output_path)
    print(f"\n训练完成！最终模型已保存到 {output_path}")

    return model
