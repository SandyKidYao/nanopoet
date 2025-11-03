"""
Reward 模型训练 - 作者风格匹配分类器

训练一个二元分类器来判断"作者-诗词"配对是否合理，作为强化学习阶段的奖励信号。

关键特性：
1. 基于预训练的 GPT 模型，添加双向 Transformer 层
2. 冻结 GPT 权重，只训练双向层和分类头
3. 使用全局平均池化而非只取最后一个 token
4. 支持 checkpoint 恢复训练
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanopoet.common import encode_poem
from nanopoet.model import GPTLanguageModel, FeedForward


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件（通过文件名自然排序）"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoint_files = sorted(checkpoint_path.glob("epoch_*.pt"))
    if not checkpoint_files:
        return None

    return checkpoint_files[-1]


# ==================== 分类器模型定义 ====================

class BidirectionalMultiHeadAttention(nn.Module):
    """双向多头注意力（无 Causal Mask）"""

    def __init__(self, emb_size, head_num, dropout=0.0):
        super().__init__()
        assert emb_size % head_num == 0

        self.emb_size = emb_size
        self.head_num = head_num
        self.head_size = emb_size // head_num

        self.qkv = nn.Linear(emb_size, 3 * emb_size, bias=False)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.emb_size, dim=-1)

        q = q.view(B, T, self.head_num, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.head_num, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.head_num, self.head_size).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        # 不做 causal masking，全连接注意力
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class BidirectionalTransformerBlock(nn.Module):
    """双向 Transformer Block"""

    def __init__(self, emb_size, head_num, dropout=0.0):
        super().__init__()
        self.sa = BidirectionalMultiHeadAttention(emb_size, head_num, dropout)
        self.ffwd = FeedForward(emb_size, dropout)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BinaryClassifier(nn.Module):
    """
    二元分类器（改进版）

    架构：
    1. GPT 编码器（causal，复用预训练权重）
    2. 双向 Transformer 层（全局感知）
    3. 全局平均池化
    4. 分类头
    """

    def __init__(self, gpt_model: GPTLanguageModel, freeze_base=True, num_bidirectional_layers=2):
        super().__init__()
        self.gpt = gpt_model

        # 冻结 GPT 参数
        if freeze_base:
            for param in self.gpt.parameters():
                param.requires_grad = False

        # 双向 Transformer 层
        self.bidirectional_blocks = nn.Sequential(*[
            BidirectionalTransformerBlock(
                emb_size=self.gpt.emb_size,
                head_num=self.gpt.head_num,
                dropout=0.1
            )
            for _ in range(num_bidirectional_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt.emb_size, self.gpt.emb_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.gpt.emb_size // 2, 1)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 阶段1：通过 GPT 编码（causal）
        tok_emb = self.gpt.token_embedding(idx)
        pos_emb = self.gpt.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.gpt.blocks(x)
        x = self.gpt.ln_f(x)

        # 阶段2：通过双向层（全局感知）
        x = self.bidirectional_blocks(x)

        # 阶段3：全局平均池化
        pooled = x.mean(dim=1)  # [B, T, D] -> [B, D]

        # 阶段4：分类
        logits = self.classifier(pooled).squeeze(-1)  # [B]

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())

        return logits if loss is None else (logits, loss)


# ==================== 数据处理函数 ====================

def create_positive_sample(poem):
    """
    创建正样本：保留原始作者
    只保留 author 和 content 字段
    """
    sample_dict = {
        "author": poem["author"],
        "content": poem["content"]
    }
    return encode_poem(sample_dict)


def create_negative_sample(poem, all_authors):
    """
    创建负样本：随机替换作者
    只保留 author 和 content 字段
    """
    wrong_authors = [a for a in all_authors if a != poem['author']]
    wrong_author = random.choice(wrong_authors)

    sample_dict = {
        "author": wrong_author,
        "content": poem["content"]
    }
    return encode_poem(sample_dict)


def generate_samples(data, all_authors):
    """
    生成训练样本（正负样本各一半）

    策略：
    1. 过滤出作者在 all_authors（AUTHOR_S）中的所有诗词
    2. 所有诗词先作为正样本（保留原作者）
    3. 所有诗词再生成负样本（随机替换作者）
    4. 合并并打乱顺序

    这样可以充分利用所有数据，正负样本数量相等
    """
    # 过滤出作者在目标列表中的诗词
    filtered_data = [p for p in data if p["author"] in all_authors]

    if len(filtered_data) == 0:
        raise ValueError(f"没有找到作者在 {all_authors} 中的诗词！")

    samples = []
    labels = []

    # 1. 生成所有正样本（保留原作者）
    for poem in filtered_data:
        sample = create_positive_sample(poem)
        samples.append(sample)
        labels.append(1)

    # 2. 生成等量负样本（随机替换作者）
    for poem in filtered_data:
        sample = create_negative_sample(poem, all_authors)
        samples.append(sample)
        labels.append(0)

    # 3. 打乱顺序（保持样本和标签对应）
    combined = list(zip(samples, labels))
    random.shuffle(combined)
    samples, labels = zip(*combined)
    samples = list(samples)
    labels = list(labels)

    return samples, labels


def get_classifier_batch(encoded_samples, labels, batch_size, block_size, device):
    """获取一个 batch 的分类数据"""
    indices = torch.randint(len(encoded_samples), (batch_size,))
    batch_samples = [encoded_samples[i] for i in indices]
    batch_labels = [labels[i] for i in indices]

    # 截断和填充
    batch_x = []
    for sample in batch_samples:
        if len(sample) > block_size:
            sample = sample[:block_size]
        batch_x.append(sample)

    # 填充到统一长度
    max_len = max(len(x) for x in batch_x)
    padded_x = []
    for x in batch_x:
        if len(x) < max_len:
            pad_len = max_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        padded_x.append(x)

    batch_x = torch.stack(padded_x).to(device)
    batch_y = torch.tensor(batch_labels, dtype=torch.long).to(device)

    return batch_x, batch_y


def estimate_loss_and_acc(encoded_samples, labels, classifier, eval_iters, batch_size, block_size, device):
    """评估损失和准确率"""
    losses = []
    accs = []

    for _ in range(eval_iters):
        X, Y = get_classifier_batch(encoded_samples, labels, batch_size, block_size, device)
        with torch.no_grad():
            logits, loss = classifier(X, Y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == Y).float().mean()

        losses.append(loss.item())
        accs.append(acc.item())

    return sum(losses) / len(losses), sum(accs) / len(accs)


# ==================== 主训练函数 ====================

def train_reward_classifier(
        gpt_model: GPTLanguageModel,
        tokenizer,
        train_poems,
        val_poems,
        device,
        batch_size=16,
        learning_rate=1e-4,
        total_epochs=10,
        eval_interval=50,
        eval_iters=10,
        checkpoint_dir="./output/checkpoints/reward",
        output_path="./output/reward_classifier.pt",
        pretrain_model_path=None,
        num_bidirectional_layers=2,
):
    """
    训练作者风格匹配分类器

    Args:
        gpt_model: 预训练的 GPT 模型（将被冻结）
        tokenizer: 分词器
        train_poems: 训练集诗词数据
        val_poems: 验证集诗词数据
        device: 设备
        batch_size: 批次大小
        learning_rate: 学习率
        total_epochs: 总训练轮数
        eval_interval: 评估间隔（步数）
        eval_iters: 评估时的迭代次数
        checkpoint_dir: checkpoint 保存目录
        output_path: 最终模型保存路径
        pretrain_model_path: GPT 预训练模型路径（通常是 SFT 的结果）
        num_bidirectional_layers: 双向层数量

    注意：样本数量由训练数据自动确定（所有符合条件的诗词都用作正样本，并生成等量负样本）
    """
    # 创建必要的目录
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型
    if pretrain_model_path:
        print(f"加载预训练模型: {pretrain_model_path}")
        pretrain_state = torch.load(pretrain_model_path, map_location=device)
        # 处理可能的字典包装
        if isinstance(pretrain_state, dict) and 'model_state_dict' in pretrain_state:
            pretrain_state = pretrain_state['model_state_dict']
        gpt_model.load_state_dict(pretrain_state, strict=True)
        print("预训练模型加载成功")

    # 将 GPT 模型移到设备
    gpt_model = gpt_model.to(device)
    gpt_model.eval()  # GPT 始终保持 eval 模式（因为被冻结）

    # 创建分类器
    print(f"\n创建二元分类器（冻结 GPT，{num_bidirectional_layers} 层双向 Transformer）...")
    classifier = BinaryClassifier(
        gpt_model=gpt_model,
        freeze_base=True,
        num_bidirectional_layers=num_bidirectional_layers
    ).to(device)

    # 统计参数
    gpt_params = sum(p.numel() for p in classifier.gpt.parameters() if p.requires_grad)
    bidirectional_params = sum(p.numel() for p in classifier.bidirectional_blocks.parameters())
    classifier_head_params = sum(p.numel() for p in classifier.classifier.parameters())
    total_trainable = gpt_params + bidirectional_params + classifier_head_params

    print(f"参数统计：")
    print(f"  - GPT 参数: {gpt_params:,} (已冻结)")
    print(f"  - 双向层参数: {bidirectional_params:,}")
    print(f"  - 分类头参数: {classifier_head_params:,}")
    print(f"  - 总可训练参数: {total_trainable:,}")

    all_authors = list(set([p["author"] for p in train_poems]))

    print(f"\n数据集统计：")
    print(f"  - 训练集诗词数: {len(train_poems)}")
    print(f"  - 验证集诗词数: {len(val_poems)}")
    print(f"  - 目标作者数量: {len(all_authors)} (来自 AUTHOR_S)")

    # 生成训练和验证样本
    print(f"\n生成训练样本...")
    print(f"  注意：只使用作者在 AUTHOR_S ({len(all_authors)}人) 中的诗词生成样本")
    print(f"  策略：所有符合条件的诗词都作为正样本，并生成等量负样本（打乱作者）")

    train_samples, train_labels = generate_samples(train_poems, all_authors)
    val_samples, val_labels = generate_samples(val_poems, all_authors)

    # 编码成 token ids
    train_encoded = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in train_samples]
    val_encoded = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in val_samples]

    avg_train_len = sum(len(x) for x in train_encoded) / len(train_encoded)
    avg_val_len = sum(len(x) for x in val_encoded) / len(val_encoded)

    print(f"  - 训练样本数: {len(train_samples)} (正负各半)")
    print(f"  - 验证样本数: {len(val_samples)} (正负各半)")
    print(f"  - 平均序列长度: 训练={avg_train_len:.1f}, 验证={avg_val_len:.1f}")

    block_size = gpt_model.block_size

    # 初始化优化器
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)

    # 尝试从最新的 checkpoint 恢复训练
    start_epoch = 0
    global_step = 0
    latest_checkpoint_file = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_file:
        print(f"\n发现checkpoint，从 {latest_checkpoint_file} 恢复训练...")
        checkpoint = torch.load(latest_checkpoint_file, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        print(f"从第 {start_epoch} 个epoch继续训练 (global_step={global_step})")
    else:
        print("\n未发现checkpoint，从头开始训练...")

    # 计算总步数
    steps_per_epoch = len(train_encoded) // batch_size
    total_steps = total_epochs * steps_per_epoch

    print(f"\n训练配置:")
    print(f"  - 总轮数: {total_epochs}")
    print(f"  - 每轮步数: {steps_per_epoch}")
    print(f"  - 总步数: {total_steps}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {learning_rate:.2e}")
    print(f"  - 评估间隔: 每 {eval_interval} 步")

    # 训练循环
    classifier.train()
    for epoch in range(start_epoch, total_epochs):
        print(f"\n===== Epoch {epoch + 1}/{total_epochs} =====")

        for step in range(steps_per_epoch):
            global_step += 1

            # 获取 batch
            xb, yb = get_classifier_batch(train_encoded, train_labels, batch_size, block_size, device)

            # 前向传播
            logits, loss = classifier(xb, yb)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 定期评估
            if global_step % eval_interval == 0 or global_step == 1:
                classifier.eval()

                train_loss, train_acc = estimate_loss_and_acc(
                    train_encoded, train_labels, classifier, eval_iters, batch_size, block_size, device
                )
                val_loss, val_acc = estimate_loss_and_acc(
                    val_encoded, val_labels, classifier, eval_iters, batch_size, block_size, device
                )

                print(f"Step {global_step:5d} | "
                      f"训练 Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                      f"验证 Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

                classifier.train()

        # 每个 epoch 结束时评估并保存 checkpoint
        classifier.eval()
        train_loss, train_acc = estimate_loss_and_acc(
            train_encoded, train_labels, classifier, eval_iters, batch_size, block_size, device
        )
        val_loss, val_acc = estimate_loss_and_acc(
            val_encoded, val_labels, classifier, eval_iters, batch_size, block_size, device
        )

        print(f"Epoch {epoch + 1} 结束 | "
              f"训练 Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"验证 Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        # 保存 checkpoint
        checkpoint_file = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': global_step,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_file)
        print(f"✓ Checkpoint已保存: {checkpoint_file}")

        classifier.train()

    # 训练完成，最终评估
    classifier.eval()
    train_loss, train_acc = estimate_loss_and_acc(
        train_encoded, train_labels, classifier, eval_iters, batch_size, block_size, device
    )
    val_loss, val_acc = estimate_loss_and_acc(
        val_encoded, val_labels, classifier, eval_iters, batch_size, block_size, device
    )

    print(f"\nFinal | "
          f"训练 Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
          f"验证 Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

    # 测试推理示例
    print("\n测试推理示例（验证作者-诗词匹配度）：")
    print("  注：预测概率越高，说明模型认为该作者与诗词风格越匹配\n")

    test_examples = [
        # 李白的诗 - 正确配对和错误配对
        {"author": "李白", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "label": "✓ 正确"},
        {"author": "杜甫", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "label": "✗ 错误"},
        {"author": "苏轼", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "label": "✗ 错误"},

        # 杜甫的诗 - 正确配对和错误配对
        {"author": "杜甫", "content": "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。", "label": "✓ 正确"},
        {"author": "李白", "content": "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。", "label": "✗ 错误"},

        # 苏轼的词 - 正确配对和错误配对
        {"author": "苏轼", "content": "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。", "label": "✓ 正确"},
        {"author": "李白", "content": "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。", "label": "✗ 错误"},
    ]

    for poem_dict in test_examples:
        sample = encode_poem({"author": poem_dict["author"], "content": poem_dict["content"]})
        test_ids = torch.tensor([tokenizer.encode(sample)], device=device)
        with torch.no_grad():
            logits = classifier(test_ids)
            prob = torch.sigmoid(logits)

        content_preview = poem_dict['content'][:20] + "..." if len(poem_dict['content']) > 20 else poem_dict['content']
        print(
            f"  {poem_dict['label']} | 作者: {poem_dict['author']:4s} | 内容: {content_preview:24s} | 预测: {prob.item():.2%}")

    # 保存最终模型
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'emb_size': gpt_model.emb_size,
        'block_size': gpt_model.block_size,
        'layer_num': gpt_model.layer_num,
        'head_num': gpt_model.head_num,
        'num_bidirectional_layers': num_bidirectional_layers,
    }, output_path)
    print(f"\n训练完成！最终模型已保存到 {output_path}")

    return classifier
