import torch
from pathlib import Path

from nanopoet.common import BEGIN
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


def get_batch(data, batch_size, block_size, device):
    """数据加载函数"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(train_data, val_data, model, eval_iters, batch_size, device):
    """评估损失函数"""
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, model.block_size, device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def train_pre(
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
):
    # 创建checkpoint目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先构建训练数据
    train_data = torch.tensor(tokenizer.encode("".join([d["content"] for d in train_poems])), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode("".join([d["content"] for d in val_poems])), dtype=torch.long)
    block_size = model.block_size

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 尝试从最新的checkpoint恢复训练
    start_epoch = 0
    global_step = 0
    latest_checkpoint_file = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_file:
        print(f"发现checkpoint，从 {latest_checkpoint_file} 恢复训练...")
        loaded_epoch, loaded_step = load_checkpoint(latest_checkpoint_file, model, optimizer)
        start_epoch = loaded_epoch + 1  # 从下一个epoch开始
        global_step = loaded_step
        print(f"从第 {start_epoch} 个epoch继续训练 (global_step={global_step})")
    else:
        print("未发现checkpoint，从头开始训练...")

    # 将模型移到设备上
    model = model.to(device)
    model.train()

    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        print(f"\n===== Epoch {epoch + 1}/{total_epochs} =====")
        # 计算本轮需要的迭代次数
        steps_per_epoch = len(train_data) // (batch_size * block_size)

        for step in range(steps_per_epoch):
            global_step += 1
            xb, yb = get_batch(train_data, batch_size, block_size, device)
            # 前向传播
            logits, loss = model(xb, yb)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # 定期评估和生成样本
            if global_step % eval_interval == 0 or global_step == 1:
                model.eval()
                out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, device)
                print(f"Step {global_step:5d} | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f}")
                print(f"Sample: {generate_sample(model, tokenizer, device, BEGIN)}")
                model.train()

        # 每个epoch结束时评估并保存checkpoint
        model.eval()
        out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, device)
        train_loss = out['train'].item()
        val_loss = out['val'].item()
        print(f"Epoch {epoch + 1} 结束 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        model.train()

        # 保存checkpoint（文件名自然有序）
        checkpoint_file = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(model, optimizer, epoch, global_step, train_loss, val_loss, checkpoint_file)

    # 训练完成，评估 + 生成样本
    model.eval()
    out = estimate_loss(train_data, val_data, model, eval_iters, batch_size, device)
    print(f"\nFinal | Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f}")
    print(f"Sample: {generate_sample(model, tokenizer, device, BEGIN)}")

    # 保存最终模型
    torch.save(model.state_dict(), output_path)
    print(f"\n训练完成！最终模型已保存到 {output_path}")

    return model
