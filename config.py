from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RunConfig:
    run_name: str
    model_type: str  # baseline | modern
    size: str  # small | medium | large
    d_model: int
    nhead: int
    enc_layers: int
    dec_layers: int
    dropout: float
    num_kv_heads: int
    ffn_mult: float
    lr_factor: float
    warmup_steps: int
    batch_size: int
    grad_accum: int
    max_tgt_len: int
    max_decode_len: int
    epochs: int
    seed: int