from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, t: int, device: torch.device, dtype: torch.dtype):
        pos = torch.arange(t, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", pos, self.inv_freq)          # (t, D/2)
        emb = torch.cat([freqs, freqs], dim=-1).to(device=device, dtype=dtype)  # (t, D)
        cos = emb.cos()[None, None, :, :]  # (1,1,t,D)
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q: (B,H,Tq,D) , k: (B,H,Tk,D)
        Tq = q.size(-2)
        Tk = k.size(-2)
        t = max(Tq, Tk)

        cos, sin = self._build_cos_sin(t, device=q.device, dtype=q.dtype)

        cos_q, sin_q = cos[:, :, :Tq, :], sin[:, :, :Tq, :]
        cos_k, sin_k = cos[:, :, :Tk, :], sin[:, :, :Tk, :]

        q2 = (q * cos_q) + (_rotate_half(q) * sin_q)
        k2 = (k * cos_k) + (_rotate_half(k) * sin_k)
        return q2, k2

class Attention(nn.Module):
    """Multi-head attention with optional Grouped-Query Attention (GQA) and RoPE."""

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads or num_q_heads
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.head_dim = d_model // num_q_heads
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RoPE(self.head_dim) if use_rope else None

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # x_q: (B, Tq, D), x_kv: (B, Tk, D)
        B, Tq, _ = x_q.shape
        Tk = x_kv.size(1)

        q = self.q_proj(x_q).view(B, Tq, self.num_q_heads, self.head_dim).transpose(1, 2)  # (B, Hq, Tq, Hd)
        k = self.k_proj(x_kv).view(B, Tk, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, Tk, Hd)
        v = self.v_proj(x_kv).view(B, Tk, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k)

        # Expand kv heads to match q heads (GQA)
        if self.num_kv_heads != self.num_q_heads:
            repeat = self.num_q_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, Tq, Tk)

        if attn_mask is not None:
            # attn_mask expected shape broadcastable to (B, H, Tq, Tk)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) where True means pad
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, Tq, Hd)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out_proj(out)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # (1,1,T,T) with -inf above diagonal
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m[None, None, :, :]