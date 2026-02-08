"""Experiment runner with tracking and parallelization."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.experiment_config import ExperimentConfig
from ..models.modern_transformer import ModernTransformer
from ..utils.training import LabelSmoothingLoss, NoamLR, decode_loader_full
from ..utils.metrics import compute_all_metrics, corpus_bleu
from ..utils.vocab import Vocab


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExperimentRunner:
    """Manages experiment execution and result tracking."""
    
    def __init__(
        self,
        output_dir: str = "runs",
        results_csv: str = "runs/results.csv",
        device: Optional[torch.device] = None,
    ):
        self.output_dir = Path(output_dir)
        self.results_csv = Path(results_csv)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
    
    def run_already_exists(self, run_name: str) -> bool:
        """Check if experiment already completed."""
        if not self.results_csv.exists():
            return False
        df = pd.read_csv(self.results_csv)
        return ((df["run_name"] == run_name) & (df["epoch"].astype(str) == "final_test")).any()
    
    def train_single_experiment(
        self,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        text_vocab: Vocab,
        gloss_vocab: Vocab,
        skip_if_exists: bool = True,
    ) -> Optional[str]:
        """Train a single experiment with given configuration."""
        
        run_name = config.get_run_name()
        
        if skip_if_exists and self.run_already_exists(run_name):
            print(f"⏭️  Skipping (already completed): {run_name}")
            return None
        
        print(f"\n🚀 Running: {run_name}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Build model
        model = ModernTransformer(
            src_vocab_size=len(text_vocab.tokens),
            tgt_vocab_size=len(gloss_vocab.tokens),
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            num_kv_heads=config.num_kv_heads,
            ffn_mult=config.ffn_mult,
            pad_id_src=text_vocab.pad_id,
            pad_id_tgt=gloss_vocab.pad_id,
            use_rope=config.use_rope,
        ).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamLR(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps, factor=config.lr_factor)
        loss_fn = LabelSmoothingLoss(
            vocab_size=len(gloss_vocab.tokens),
            padding_idx=gloss_vocab.pad_id,
            smoothing=config.label_smoothing
        )
        
        # Training loop
        best_val_loss = float("inf")
        ckpt_path = self.output_dir / f"{run_name}_best.pt"
        
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
            for step, batch in enumerate(pbar, start=1):
                src = batch.src.to(self.device)
                tgt = batch.tgt.to(self.device)
                src_kpm = batch.src_key_padding_mask.to(self.device)
                tgt_kpm = batch.tgt_key_padding_mask.to(self.device)
                
                logits = model(src, tgt, src_kpm, tgt_kpm)
                gold = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
                (loss / config.grad_accum).backward()
                
                if step % config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr = scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr = scheduler.last_lr
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{total_loss/step:.4f}", "lr": f"{lr:.2e}"})
            
            # Validation
            val_loss = self._eval_loss(model, val_loader, loss_fn)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "config": config.to_dict()}, ckpt_path)
            
            # Quick BLEU on validation
            val_bleu = self._eval_bleu_sample(model, val_loader, text_vocab, gloss_vocab, config, max_samples=200)
            
            print(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}, val_bleu={val_bleu:.2f}")
        
        # Final test evaluation
        ckpt_obj = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt_obj["model"])
        
        test_preds, test_refs = decode_loader_full(
            model=model,
            loader=test_loader,
            text_vocab=text_vocab,
            gloss_vocab=gloss_vocab,
            max_decode_len=config.max_decode_len,
            device=self.device,
        )
        
        test_metrics = compute_all_metrics(test_preds, test_refs)
        
        # Log results
        final_row = {
            **config.to_dict(),
            "run_name": run_name,
            "epoch": "final_test",
            "checkpoint": str(ckpt_path),
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "test_size": len(test_refs),
        }
        
        self._append_csv(final_row)
        
        print(f"✅ Completed: {run_name}")
        print(f"   BLEU-4: {test_metrics['bleu4']:.2f}")
        print(f"   Checkpoint: {ckpt_path}")
        
        return str(ckpt_path)
    
    def _eval_loss(self, model, loader, loss_fn):
        """Evaluate loss on a dataset."""
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                src = batch.src.to(self.device)
                tgt = batch.tgt.to(self.device)
                src_kpm = batch.src_key_padding_mask.to(self.device)
                tgt_kpm = batch.tgt_key_padding_mask.to(self.device)
                logits = model(src, tgt, src_kpm, tgt_kpm)
                gold = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
                total += float(loss.item())
                n += 1
        return total / max(1, n)
    
    def _eval_bleu_sample(self, model, loader, text_vocab, gloss_vocab, config, max_samples=200):
        """Evaluate BLEU on a sample of the dataset."""
        model.eval()
        preds = []
        refs = []
        seen = 0
        
        from ..utils.training import greedy_decode
        from ..utils.vocab import decode
        
        with torch.no_grad():
            for batch in loader:
                src = batch.src.to(self.device)
                src_kpm = batch.src_key_padding_mask.to(self.device)
                tgt = batch.tgt.to(self.device)
                
                ys = greedy_decode(
                    model,
                    src=src,
                    src_key_padding_mask=src_kpm,
                    bos_id=gloss_vocab.bos_id,
                    eos_id=gloss_vocab.eos_id,
                    pad_id=gloss_vocab.pad_id,
                    max_len=config.max_decode_len,
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
                    seen += 1
                    if seen >= max_samples:
                        break
                if seen >= max_samples:
                    break
        
        if not preds:
            return 0.0
        return corpus_bleu(preds, refs)
    
    def _append_csv(self, row: Dict):
        """Append results to CSV file."""
        file_exists = self.results_csv.exists()
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)
    
    def run_experiments(
        self,
        configs: List[ExperimentConfig],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        text_vocab: Vocab,
        gloss_vocab: Vocab,
        skip_if_exists: bool = True,
    ) -> Dict[str, str]:
        """Run multiple experiments sequentially."""
        checkpoints = {}
        
        for config in configs:
            ckpt = self.train_single_experiment(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                text_vocab=text_vocab,
                gloss_vocab=gloss_vocab,
                skip_if_exists=skip_if_exists,
            )
            if ckpt:
                checkpoints[config.get_run_name()] = ckpt
        
        print(f"\n✅ All experiments completed!")
        print(f"Results saved to: {self.results_csv}")
        
        return checkpoints
