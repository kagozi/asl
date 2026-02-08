"""Training utilities: loss functions, schedulers, and inference."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .vocab import Vocab, decode


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, V)
        target: (N,)
        """
        log_probs = logits.log_softmax(dim=-1)

        mask = target.ne(self.padding_idx)
        if mask.sum() == 0:
            return log_probs.sum() * 0.0

        log_probs = log_probs[mask]
        target = target[mask]

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0.0

        return -(true_dist * log_probs).sum(dim=-1).mean()


class NoamLR:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        self.last_lr = 0.0

    def step(self):
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5))))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.last_lr = lr
        return lr


def greedy_decode(
    model,
    src: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 100,
) -> torch.Tensor:
    """Greedy decoding for ModernTransformer."""
    device = src.device
    B = src.size(0)

    memory = model.encode(src, src_key_padding_mask)
    
    def decode_step(ys):
        return model.decode(ys, memory, tgt_key_padding_mask=None, memory_key_padding_mask=src_key_padding_mask)

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        dec_out = decode_step(ys)
        logits = model.lm_head(dec_out[:, -1, :])
        next_id = logits.argmax(dim=-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        finished |= next_id.eq(eos_id)
        if finished.all():
            break

    return ys


def decode_loader_full(
    model,
    loader: DataLoader,
    text_vocab: Vocab,
    gloss_vocab: Vocab,
    max_decode_len: int,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """Decode all items in loader, returns (pred_strs, ref_strs)."""
    model.eval()
    preds, refs = [], []

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_kpm = batch.src_key_padding_mask.to(device)
            tgt = batch.tgt.to(device)

            ys = greedy_decode(
                model,
                src=src,
                src_key_padding_mask=src_kpm,
                bos_id=gloss_vocab.bos_id,
                eos_id=gloss_vocab.eos_id,
                pad_id=gloss_vocab.pad_id,
                max_len=max_decode_len,
            )

            for i in range(src.size(0)):
                hyp_ids = ys[i].tolist()
                if gloss_vocab.eos_id in hyp_ids:
                    hyp_ids = hyp_ids[: hyp_ids.index(gloss_vocab.eos_id) + 1]
                hyp = decode(hyp_ids, gloss_vocab, skip_special=True)

                ref_ids = tgt[i].tolist()
                ref = decode(ref_ids, gloss_vocab, skip_special=True)

                preds.append(hyp)
                refs.append(ref)

    return preds, refs
