# ============================================================================
# evaluation/metrics.py
# ============================================================================
"""Evaluation metrics for machine translation."""

import torch
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

# Import evaluation libraries
try:
    from sacrebleu import corpus_bleu, BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

try:
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. Install with: pip install rouge-score")


class TranslationMetrics:
    """Comprehensive evaluation metrics for translation."""
    
    def __init__(self):
        """Initialize metrics calculators."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        ) if ROUGE_AVAILABLE else None
        
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    def compute_bleu(
        self, 
        references: List[List[str]], 
        hypotheses: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            references: List of reference translations (each is a list of tokens)
            hypotheses: List of hypothesis translations (each is a list of tokens)
        
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        if not NLTK_AVAILABLE:
            return {'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0}
        
        # Wrap each reference in a list (NLTK expects multiple references per hypothesis)
        references_wrapped = [[ref] for ref in references]
        
        bleu_scores = {}
        
        # Compute individual BLEU-n scores
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            try:
                score = nltk_corpus_bleu(
                    references_wrapped,
                    hypotheses,
                    weights=weights,
                    smoothing_function=self.smoothing
                )
                bleu_scores[f'bleu-{n}'] = score * 100  # Convert to percentage
            except Exception as e:
                print(f"Error computing BLEU-{n}: {e}")
                bleu_scores[f'bleu-{n}'] = 0.0
        
        return bleu_scores
    
    def compute_sacrebleu(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute SacreBLEU score (more standardized).
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            BLEU score (0-100)
        """
        if not SACREBLEU_AVAILABLE:
            return 0.0
        
        try:
            # SacreBLEU expects list of references for each hypothesis
            refs = [[ref] for ref in references]
            bleu = corpus_bleu(hypotheses, refs)
            return bleu.score
        except Exception as e:
            print(f"Error computing SacreBLEU: {e}")
            return 0.0
    
    def compute_meteor(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            METEOR score (0-1)
        """
        if not NLTK_AVAILABLE:
            return 0.0
        
        try:
            scores = []
            for ref, hyp in zip(references, hypotheses):
                score = meteor_score([ref.split()], hyp.split())
                scores.append(score)
            return np.mean(scores) * 100  # Convert to percentage
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            return 0.0
    
    def compute_rouge(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        try:
            rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
            for ref, hyp in zip(references, hypotheses):
                scores = self.rouge_scorer.score(ref, hyp)
                rouge_scores['rouge-1'] += scores['rouge1'].fmeasure
                rouge_scores['rouge-2'] += scores['rouge2'].fmeasure
                rouge_scores['rouge-l'] += scores['rougeL'].fmeasure
            
            # Average and convert to percentage
            n = len(references)
            rouge_scores = {k: (v / n) * 100 for k, v in rouge_scores.items()}
            
            return rouge_scores
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    def compute_chrf(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute chrF++ score (character n-gram F-score).
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            chrF++ score (0-100)
        """
        try:
            from sacrebleu import corpus_chrf
            refs = [[ref] for ref in references]
            chrf = corpus_chrf(hypotheses, refs)
            return chrf.score
        except ImportError:
            # Fallback: simple character-level F1
            return self._simple_chrf(references, hypotheses)
        except Exception as e:
            print(f"Error computing chrF: {e}")
            return 0.0
    
    def _simple_chrf(self, references: List[str], hypotheses: List[str]) -> float:
        """Simple character-level F1 score."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_chars = set(ref)
            hyp_chars = set(hyp)
            
            if len(hyp_chars) == 0:
                scores.append(0.0)
                continue
            
            precision = len(ref_chars & hyp_chars) / len(hyp_chars)
            recall = len(ref_chars & hyp_chars) / len(ref_chars) if len(ref_chars) > 0 else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                scores.append(f1)
            else:
                scores.append(0.0)
        
        return np.mean(scores) * 100
    
    def compute_all_metrics(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        # Tokenize for BLEU
        refs_tokens = [ref.split() for ref in references]
        hyps_tokens = [hyp.split() for hyp in hypotheses]
        
        # BLEU scores
        bleu_scores = self.compute_bleu(refs_tokens, hyps_tokens)
        metrics.update(bleu_scores)
        
        # SacreBLEU
        if SACREBLEU_AVAILABLE:
            metrics['sacrebleu'] = self.compute_sacrebleu(references, hypotheses)
        
        # METEOR
        if NLTK_AVAILABLE:
            metrics['meteor'] = self.compute_meteor(references, hypotheses)
        
        # ROUGE
        if ROUGE_AVAILABLE:
            rouge_scores = self.compute_rouge(references, hypotheses)
            metrics.update(rouge_scores)
        
        # chrF++
        metrics['chrf++'] = self.compute_chrf(references, hypotheses)
        
        return metrics