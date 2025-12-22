# ============================================================================
# evaluation/evaluator.py
# ============================================================================
"""Model evaluator for translation tasks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
from pathlib import Path
import json
import pandas as pd

from .metrics import TranslationMetrics


class TranslationEvaluator:
    """Evaluator for translation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        src_vocab,
        tgt_vocab,
        max_gen_len: int = 100,
        beam_size: int = 1
    ):
        """
        Args:
            model: Translation model
            device: Device to run evaluation on
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            max_gen_len: Maximum generation length
            beam_size: Beam size for beam search (1 = greedy)
        """
        self.model = model.to(device)
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_gen_len = max_gen_len
        self.beam_size = beam_size
        
        self.metrics_calculator = TranslationMetrics()
        
        # Special token indices
        self.start_idx = tgt_vocab.word2idx.get('<start>', 2)
        self.end_idx = tgt_vocab.word2idx.get('<end>', 3)
        self.pad_idx = tgt_vocab.word2idx.get('<pad>', 0)
    
    @torch.no_grad()
    def generate_translation(self, src: torch.Tensor) -> List[int]:
        """
        Generate translation for a single source sequence.
        
        Args:
            src: Source tensor (1, src_len)
        
        Returns:
            List of predicted token indices
        """
        self.model.eval()
        src = src.to(self.device)
        
        # Use model's generate method
        if hasattr(self.model, 'generate'):
            output = self.model.generate(
                src,
                max_len=self.max_gen_len,
                start_token_idx=self.start_idx,
                end_token_idx=self.end_idx
            )
            # Convert to list and remove start token
            pred_indices = output[0, 1:].cpu().tolist()
        else:
            # Fallback greedy decoding
            pred_indices = self._greedy_decode(src)
        
        return pred_indices
    
    def _greedy_decode(self, src: torch.Tensor) -> List[int]:
        """Greedy decoding fallback."""
        ys = torch.tensor([[self.start_idx]], device=self.device)
        
        for _ in range(self.max_gen_len):
            # Create dummy target with proper shape
            tgt = torch.cat([ys, torch.zeros(1, 1, dtype=torch.long, device=self.device)], dim=1)
            
            # Forward pass
            logits = self.model(src, tgt)
            
            # Get next token
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            
            if next_token.item() == self.end_idx:
                break
            
            ys = torch.cat([ys, next_token], dim=1)
        
        return ys[0, 1:].cpu().tolist()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        num_examples: Optional[int] = None,
        save_predictions: bool = False,
        output_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            num_examples: Number of examples to evaluate (None = all)
            save_predictions: Whether to save predictions
            output_path: Path to save predictions
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        references = []
        hypotheses = []
        sources = []
        
        print(f"Evaluating model...")
        progress_bar = tqdm(test_loader, desc='Generating translations')
        
        total_examples = 0
        for src_batch, tgt_batch in progress_bar:
            batch_size = src_batch.size(0)
            
            for i in range(batch_size):
                if num_examples and total_examples >= num_examples:
                    break
                
                # Get source and target
                src = src_batch[i:i+1]
                tgt = tgt_batch[i]
                
                # Generate prediction
                pred_indices = self.generate_translation(src)
                
                # Decode to text
                src_text = self.src_vocab.decode(src[0].cpu().tolist(), skip_special_tokens=True)
                ref_text = self.tgt_vocab.decode(tgt.cpu().tolist(), skip_special_tokens=True)
                pred_text = self.tgt_vocab.decode(pred_indices, skip_special_tokens=True)
                
                sources.append(src_text)
                references.append(ref_text)
                hypotheses.append(pred_text)
                
                total_examples += 1
                
                # Update progress
                if total_examples % 10 == 0:
                    progress_bar.set_postfix({'examples': total_examples})
            
            if num_examples and total_examples >= num_examples:
                break
        
        print(f"\nEvaluated {total_examples} examples")
        
        # Compute metrics
        print("Computing metrics...")
        metrics = self.metrics_calculator.compute_all_metrics(references, hypotheses)
        
        # Save predictions if requested
        if save_predictions and output_path:
            self._save_predictions(sources, references, hypotheses, metrics, output_path)
        
        return metrics
    
    def _save_predictions(
        self,
        sources: List[str],
        references: List[str],
        hypotheses: List[str],
        metrics: Dict[str, float],
        output_path: Path
    ):
        """Save predictions to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        data = {
            'metrics': metrics,
            'num_examples': len(sources),
            'predictions': [
                {
                    'source': src,
                    'reference': ref,
                    'hypothesis': hyp
                }
                for src, ref, hyp in zip(sources, references, hypotheses)
            ]
        }
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved predictions to {json_path}")
        
        # Also save as CSV for easy viewing
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame({
            'source': sources,
            'reference': references,
            'hypothesis': hypotheses
        })
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved predictions to {csv_path}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a nice format."""
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # Group metrics
        bleu_metrics = {k: v for k, v in metrics.items() if 'bleu' in k.lower()}
        rouge_metrics = {k: v for k, v in metrics.items() if 'rouge' in k.lower()}
        other_metrics = {k: v for k, v in metrics.items() if k not in bleu_metrics and k not in rouge_metrics}
        
        # Print BLEU
        if bleu_metrics:
            print("\nBLEU Scores:")
            for k, v in bleu_metrics.items():
                print(f"  {k.upper()}: {v:.2f}")
        
        # Print ROUGE
        if rouge_metrics:
            print("\nROUGE Scores:")
            for k, v in rouge_metrics.items():
                print(f"  {k.upper()}: {v:.2f}")
        
        # Print others
        if other_metrics:
            print("\nOther Metrics:")
            for k, v in other_metrics.items():
                print(f"  {k.upper()}: {v:.2f}")
        
        print("="*80 + "\n")
    
    def show_examples(
        self,
        test_loader: DataLoader,
        num_examples: int = 5
    ):
        """Show example translations."""
        self.model.eval()
        
        print("\n" + "="*80)
        print(f"EXAMPLE TRANSLATIONS (n={num_examples})")
        print("="*80 + "\n")
        
        count = 0
        for src_batch, tgt_batch in test_loader:
            batch_size = src_batch.size(0)
            
            for i in range(batch_size):
                if count >= num_examples:
                    return
                
                src = src_batch[i:i+1]
                tgt = tgt_batch[i]
                
                # Generate prediction
                pred_indices = self.generate_translation(src)
                
                # Decode
                src_text = self.src_vocab.decode(src[0].cpu().tolist(), skip_special_tokens=True)
                ref_text = self.tgt_vocab.decode(tgt.cpu().tolist(), skip_special_tokens=True)
                pred_text = self.tgt_vocab.decode(pred_indices, skip_special_tokens=True)
                
                # Print
                print(f"Example {count + 1}:")
                print(f"  Source:     {src_text}")
                print(f"  Reference:  {ref_text}")
                print(f"  Prediction: {pred_text}")
                print()
                
                count += 1
            
            if count >= num_examples:
                break