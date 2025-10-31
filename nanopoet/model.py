import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """前馈层"""

    def __init__(self, emb_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ParallelMultiHeadAttention(nn.Module):
    """
    并行化的多头注意力

    关键优化：
    1. 一次性计算所有头的 Q、K、V
    2. 使用 reshape + transpose 实现多头
    3. 并行计算所有头的注意力
    """

    def __init__(self, emb_size, head_num, block_size, dropout=0.0):
        super().__init__()
        assert emb_size % head_num == 0, "emb_size 必须能被 head_num 整除"

        self.emb_size = emb_size
        self.head_num = head_num
        self.head_size = emb_size // head_num

        # ✅ 优化 1: 一次性计算所有头的 Q、K、V
        # 传统方式：每个头 3 次 Linear = head_num * 3 次操作
        # 优化方式：1 次大的 Linear = 1 次操作
        self.qkv = nn.Linear(emb_size, 3 * emb_size, bias=False)

        # 输出投影
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels

        # ✅ 优化 2: 一次性计算 Q、K、V
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.emb_size, dim=-1)  # 每个都是 (B, T, C)

        # ✅ 优化 3: Reshape 成多头格式
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q = q.view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (B, heads, T, hs)
        k = k.view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (B, heads, T, hs)
        v = v.view(B, T, self.head_num, self.head_size).transpose(1, 2)  # (B, heads, T, hs)

        # ✅ 优化 4: 并行计算所有头的注意力
        # (B, heads, T, hs) @ (B, heads, hs, T) -> (B, heads, T, T)
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)

        # Causal masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # (B, heads, T, T) @ (B, heads, T, hs) -> (B, heads, T, hs)
        out = wei @ v

        # ✅ 优化 5: Concat 所有头
        # (B, heads, T, hs) -> (B, T, heads, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        out = self.proj(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, emb_size, head_num, block_size, dropout=0.0):
        super().__init__()
        self.sa = ParallelMultiHeadAttention(emb_size, head_num, block_size, dropout)
        self.ffwd = FeedForward(emb_size, dropout)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT 语言模型"""

    def __init__(self, vocab_size, emb_size, block_size, layer_num, head_num, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(block_size, emb_size)

        # ✅ 使用并行版本的 Transformer Block
        self.blocks = nn.Sequential(*[
            TransformerBlock(emb_size, head_num, block_size, dropout)
            for _ in range(layer_num)
        ])

        self.ln_f = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_ids=None):
        """
        生成文本

        Args:
            idx: 初始 token IDs (B, T)
            max_new_tokens: 最多生成多少个 token
            temperature: 温度参数
            top_k: top-k 采样
            stop_token_ids: 停止 token 序列（如 [60, 61, 62, 63, 64] 代表 '<EOP>'）

        Returns:
            生成的完整序列 (B, T+generated)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # 检查是否生成了停止 token 序列
            if stop_token_ids is not None and len(stop_token_ids) > 0:
                # 检查最后 N 个 token 是否匹配停止序列
                if idx.shape[1] >= len(stop_token_ids):
                    last_tokens = idx[0, -len(stop_token_ids):].tolist()
                    if last_tokens == stop_token_ids:
                        break

        return idx


def save_checkpoint(model, optimizer, epoch, step, train_loss, val_loss, checkpoint_path):
    """保存训练checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'model_state_dict': model.state_dict(),
    }
    if optimizer:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if epoch:
        data['epoch'] = epoch
    if step:
        data['step'] = step
    if train_loss:
        data['train_loss'] = train_loss
    if val_loss:
        data['val_loss'] = val_loss

    torch.save(data, checkpoint_path)

    print(f"✓ Checkpoint已保存: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载训练checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  未找到Checkpoint: {checkpoint_path}")
        return 0, 0

    print(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    train_loss = checkpoint.get('train_loss', 0)
    val_loss = checkpoint.get('val_loss', 0)

    print(f"✓ Checkpoint已加载: epoch={epoch}, step={step}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    return epoch, step


def generate_sample(model, tokenizer, device, prompt, max_tokens=150, temperature=0.8, top_k=10, end_token_ids=None):
    """生成样本"""
    # 编码prompt
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    # 生成
    generated = model.generate(
        context,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=end_token_ids
    )
    return tokenizer.decode(generated[0].tolist())
