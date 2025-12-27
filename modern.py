from __future__ import annotations

import torch
import torch.nn as nn
from common import RMSNorm, SwiGLU, RoPE, Attention

class ModernEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_kv_heads: int | None, dropout: float, ffn_mult: float = 2.7, use_rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, num_q_heads=nhead, num_kv_heads=num_kv_heads, dropout=dropout, use_rope=use_rope)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        hidden_dim = int(d_model * ffn_mult)
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None):
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, attn_mask=None, key_padding_mask=src_key_padding_mask))
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class ModernDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_kv_heads: int | None, dropout: float, ffn_mult: float = 2.7, use_rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = Attention(d_model, num_q_heads=nhead, num_kv_heads=num_kv_heads, dropout=dropout, use_rope=use_rope)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.cross_attn = Attention(d_model, num_q_heads=nhead, num_kv_heads=num_kv_heads, dropout=dropout, use_rope=use_rope)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        hidden_dim = int(d_model * ffn_mult)
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_key_padding_mask: torch.Tensor | None, memory_key_padding_mask: torch.Tensor | None):
        # causal self-attention
        h = self.norm1(x)
        cm = causal_mask(h.size(1), device=h.device)
        x = x + self.drop1(self.self_attn(h, h, attn_mask=cm, key_padding_mask=tgt_key_padding_mask))

        # cross-attention
        h = self.norm2(x)
        x = x + self.drop2(self.cross_attn(h, memory, attn_mask=None, key_padding_mask=memory_key_padding_mask))

        # ffn
        h = self.norm3(x)
        x = x + self.drop3(self.ffn(h))
        return x


class ModernTransformer(nn.Module):
    """Small, T5-ish modernized Transformer stack (RoPE+GQA+RMSNorm+SwiGLU).

    Still token-embedding based (word-level) for the baseline stage.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 4096,
        num_kv_heads: int | None = 4,
        ffn_mult: float = 2.7,
        pad_id_src: int = 0,
        pad_id_tgt: int = 0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id_src)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id_tgt)
        self.drop = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            ModernEncoderLayer(d_model, nhead, num_kv_heads, dropout, ffn_mult=ffn_mult, use_rope=use_rope)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            ModernDecoderLayer(d_model, nhead, num_kv_heads, dropout, ffn_mult=ffn_mult, use_rope=use_rope)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # init
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=d_model ** -0.5)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None):
        x = self.drop(self.src_emb(src))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.encoder_norm(x)

    def decode(self, tgt_inp: torch.Tensor, memory: torch.Tensor, tgt_key_padding_mask: torch.Tensor | None, memory_key_padding_mask: torch.Tensor | None):
        x = self.drop(self.tgt_emb(tgt_inp))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor):
        tgt_inp = tgt[:, :-1]
        mem = self.encode(src, src_key_padding_mask)
        dec = self.decode(tgt_inp, mem, tgt_key_padding_mask=tgt_key_padding_mask[:, :-1], memory_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(dec)
        return logits