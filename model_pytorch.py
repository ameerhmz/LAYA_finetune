"""
PyTorch implementation of LAYA DecisionModel (ModernBERT-based Decision Engine).
Maintains 100% parameter name, shape, and key parity with the MLX checkpoint
(aac6fef/laya-multilingual-mlx) for zero-conversion interoperability between
NVIDIA H100 (PyTorch / CUDA / BF16) and Apple Silicon (MLX).
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(x: torch.Tensor, base: float = 160000.0) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to query or key tensor.
    Shape: [batch, heads, seq_len, head_dim]
    """
    b, h, s, d = x.shape
    device, dtype = x.device, x.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    pos = torch.arange(s, device=device).float()
    sinusoid = torch.outer(pos, inv_freq)
    sin = sinusoid.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    cos = sinusoid.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos.repeat_interleave(2, dim=-1) + rot * sin.repeat_interleave(2, dim=-1)


class MLXSequential(nn.Module):
    """Sequential container that uses .layers attribute to match MLX checkpoint keys."""
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int = 256000, hidden_size: int = 768, norm_eps: float = 1e-5):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(ids))


class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 1152):
        super().__init__()
        self.Wi = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(val) * gate)


class EncoderAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, base: float = 160000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.base = base
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.Wqkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary_emb(q, base=self.base)
        k = apply_rotary_emb(k, base=self.base)
        # Scaled Dot-Product Attention natively triggers FlashAttention-2 on Hopper H100
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.Wo(out.transpose(1, 2).reshape(b, s, -1))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        index: int,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 1152,
        norm_eps: float = 1e-5,
        is_full: bool = True,
    ):
        super().__init__()
        self.attn_norm = (
            nn.Identity() if index == 0 else nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)
        )
        self.attn = EncoderAttention(hidden_size, num_heads, base=160000.0 if is_full else 10000.0)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)
        self.mlp = EncoderMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBert(nn.Module):
    def __init__(
        self,
        num_layers: int = 22,
        vocab_size: int = 256000,
        hidden_size: int = 768,
        intermediate_size: int = 1152,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, norm_eps)
        layer_types = [
            "full_attention" if i % 3 == 0 else "sliding_attention" for i in range(num_layers)
        ]
        self.layers = nn.ModuleList([
            EncoderLayer(
                i,
                hidden_size=hidden_size,
                num_heads=12,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                is_full=(layer_types[i] == "full_attention"),
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embeddings(ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, dims: int = 768):
        super().__init__()
        self.num_heads = max(1, dims // 64)
        self.head_dim = dims // self.num_heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.in_proj(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(b, s, -1))


class HeadLayer(nn.Module):
    def __init__(self, dims: int = 768):
        super().__init__()
        self.self_attn = HeadAttention(dims)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecisionHead(nn.Module):
    def __init__(self, dims: int = 768, count: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([HeadLayer(dims) for _ in range(count)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecisionModel(nn.Module):
    """Full LAYA Decision Model (ModernBERT + DecisionHead + Scorer + ActHead)."""
    def __init__(self, dims: int = 768, num_layers: int = 22, head_layers: int = 2, num_act: int = 2):
        super().__init__()
        self.dims = dims
        self.encoder = ModernBert(num_layers=num_layers, hidden_size=dims)
        self.head = DecisionHead(dims, head_layers)
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = MLXSequential(
            nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)
        )
        self.act_head = MLXSequential(
            nn.Linear(dims + 4, 256),
            nn.GELU(),
            nn.Linear(256, num_act),
        )
        self.register_buffer("temperature", torch.ones(3, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing decision logits and action logits.
        
        Args:
            input_ids: [B, S] token ids
            attention_mask: [B, S] bool or float mask (True = valid token)
            marker_pos: [B, K] index positions of [MASK] markers in sequence
            marker_mask: [B, K] bool mask (True = valid option slot)
            qtype: [B] integer type (0=choice, 1=score, 2=noul)
            
        Returns:
            logits: [B, K] uncalibrated decision logits (masked slots set to -1e4)
            action: [B, num_act] action head logits
        """
        b, s = input_ids.shape

        # Construct 4D additive attention mask for SDPA: [B, 1, 1, S]
        # In PyTorch SDPA with bool attn_mask: True means attend, False means ignore
        attn_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)

        # 1. ModernBERT encoder
        h = self.encoder(input_ids, mask=attn_mask_4d)

        # 2. Add question type embedding [B, 1, D]
        type_embed = self.type_emb(qtype).unsqueeze(1)
        h = h + type_embed

        # 3. Decision Transformer Head
        h = self.head(h, mask=attn_mask_4d)

        # 4. Gather marker representations at [MASK] token positions
        batch_idx = torch.arange(b, device=input_ids.device).unsqueeze(1)
        clamped_pos = marker_pos.clamp(min=0, max=s - 1)
        markers = h[batch_idx, clamped_pos]  # [B, K, D]

        # 5. Score option markers
        logits = self.scorer(markers).squeeze(-1).float()  # [B, K]
        logits = torch.where(marker_mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))

        # 6. Action head features (entropy, top-2 gap, marker count)
        p = F.softmax(logits, dim=-1)
        k = marker_mask.sum(dim=-1, keepdim=True).clamp(min=2).float()
        entropy = -(p * torch.log(p.clamp(min=1e-9))).sum(dim=-1, keepdim=True) / torch.log(k)
        top = torch.topk(p, k=2, dim=-1, largest=True, sorted=True)[0]
        top1 = top[:, 0:1]
        top_diff = top[:, 0:1] - top[:, 1:2]
        features = torch.cat([top1, top_diff, entropy, k / 255.0], dim=-1)  # [B, 4]

        # Pooled CLS token representation + features
        pooled = torch.cat([h[:, 0].float(), features], dim=-1)
        action = self.act_head(pooled.to(h.dtype)).float()

        return logits, action
