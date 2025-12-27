import csv
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import asdict, dataclass
from metrics import compute_all_metrics, corpus_bleu
from vocab import Vocab
from decode import decode_loader_full, greedy_decode
from lr_scheduler import NoamLR
from loss import LabelSmoothingLoss
from modern import decode

    
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


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_run(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,   
    text_vocab: Vocab,
    gloss_vocab: Vocab,
    cfg: RunConfig,
    device: torch.device,
    out_dir: str,
    results_csv: str,
    bleu_eval_samples: int = 200,
):
    set_seed(cfg.seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps, factor=cfg.lr_factor)

    loss_fn = LabelSmoothingLoss(vocab_size=len(gloss_vocab.tokens), padding_idx=gloss_vocab.pad_id, smoothing=0.1)

    best_val_loss = float("inf")
    ckpt_path = Path(out_dir) / f"{cfg.run_name}_best.pt"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)
            src_kpm = batch.src_key_padding_mask.to(device)
            tgt_kpm = batch.tgt_key_padding_mask.to(device)

            logits = model(src, tgt, src_kpm, tgt_kpm)  # (B, T-1, V)
            # targets are next tokens
            gold = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            (loss / cfg.grad_accum).backward()

            if step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                lr = scheduler.last_lr

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/step:.4f}", "lr": f"{lr:.2e}"})

        # Validation loss
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

        # Quick BLEU (proposal-friendly; you can increase samples)
        bleu = eval_bleu(model, val_loader, text_vocab, gloss_vocab, cfg, device, max_samples=bleu_eval_samples)

    ckpt_obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_obj["model"])
    
    test_preds, test_refs = decode_loader_full(
        model=model,
        loader=test_loader,
        text_vocab=text_vocab,
        gloss_vocab=gloss_vocab,
        cfg=cfg,
        device=device,
    )
    
    test_metrics = compute_all_metrics(test_preds, test_refs)
    
    # Log a final row into CSV (run-level summary)
    final_row = {
        **asdict(cfg),
        "epoch": "final_test",
        "checkpoint": str(ckpt_path),
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "test_size": len(test_refs),
    }
    
    append_csv(results_csv, final_row)
    
    return str(ckpt_path)


def eval_loss(model, loader: DataLoader, loss_fn: LabelSmoothingLoss, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)
            src_kpm = batch.src_key_padding_mask.to(device)
            tgt_kpm = batch.tgt_key_padding_mask.to(device)
            logits = model(src, tgt, src_kpm, tgt_kpm)
            gold = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            total += float(loss.item())
            n += 1
    return total / max(1, n)


def eval_bleu(model, loader: DataLoader, text_vocab: Vocab, gloss_vocab: Vocab, cfg: RunConfig, device: torch.device, max_samples: int = 200) -> float:
    model.eval()
    preds = []
    refs = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_kpm = batch.src_key_padding_mask.to(device)
            tgt = batch.tgt.to(device)
            # decode
            ys = greedy_decode(
                model,
                src=src,
                src_key_padding_mask=src_kpm,
                bos_id=gloss_vocab.bos_id,
                eos_id=gloss_vocab.eos_id,
                pad_id=gloss_vocab.pad_id,
                max_len=cfg.max_decode_len,
            )
            for i in range(src.size(0)):
                hyp_ids = ys[i].tolist()
                # truncate at eos
                if gloss_vocab.eos_id in hyp_ids:
                    hyp_ids = hyp_ids[: hyp_ids.index(gloss_vocab.eos_id) + 1]
                hyp = decode(hyp_ids, gloss_vocab, skip_special=True)

                ref_ids = tgt[i].tolist()
                ref = decode(ref_ids, gloss_vocab, skip_special=True)

                preds.append(hyp)
                refs.append(ref)
                seen += 1
                if seen >= max_samples:
                    break
            if seen >= max_samples:
                break

    if not preds:
        return 0.0
    return corpus_bleu(preds, refs)


def append_csv(path: str, row: Dict):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)