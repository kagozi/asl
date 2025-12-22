# ============================================================================
# data/preprocessing.py
# ============================================================================
"""Data preprocessing utilities for text-gloss translation."""

import re
import string
from typing import Dict, List, Tuple
from collections import Counter
import pandas as pd


class TextGlossPreprocessor:
    """Preprocessing pipeline for text and gloss data."""
    
    def __init__(
        self,
        lowercase_text: bool = True,
        uppercase_gloss: bool = True,
        remove_punctuation: bool = True,
        remove_digits: bool = True
    ):
        self.lowercase_text = lowercase_text
        self.uppercase_gloss = uppercase_gloss
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
    
    def remove_noise(self, text: str) -> str:
        """Remove digits, punctuation, and extra whitespace."""
        if self.remove_digits:
            text = re.sub(r'\d+', '', text)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        text = " ".join(text.split())  # Normalize whitespace
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess natural language text."""
        if self.lowercase_text:
            text = text.lower()
        text = self.remove_noise(text)
        return text
    
    def preprocess_gloss(self, gloss: str) -> str:
        """Preprocess gloss notation."""
        if self.uppercase_gloss:
            gloss = gloss.upper()
        gloss = self.remove_noise(gloss)
        return gloss
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe."""
        df = df.copy()
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df['processed_gloss'] = df['gloss'].apply(self.preprocess_gloss)
        return df


class Vocabulary:
    """Vocabulary builder and manager."""
    
    SPECIAL_TOKENS = {
        'pad': '<pad>',
        'unk': '<unk>',
        'start': '<start>',
        'end': '<end>'
    }
    
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
    
    def build_from_sentences(self, sentences: List[str]):
        """Build vocabulary from list of sentences."""
        for sentence in sentences:
            self.word_counts.update(sentence.split())
        
        # Filter by minimum frequency
        filtered_words = [
            word for word, count in self.word_counts.items()
            if count >= self.min_freq
        ]
        
        # Build mappings with special tokens first
        vocab = [self.SPECIAL_TOKENS['pad'], self.SPECIAL_TOKENS['unk']]
        
        # Add start/end tokens for target vocabulary
        if self.SPECIAL_TOKENS['start'] not in vocab:
            vocab.extend([self.SPECIAL_TOKENS['start'], self.SPECIAL_TOKENS['end']])
        
        # Add regular words (sorted for consistency)
        vocab.extend(sorted(filtered_words))
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
    
    def encode(self, sentence: str, add_special_tokens: bool = False) -> List[int]:
        """Convert sentence to indices."""
        tokens = sentence.split()
        indices = [
            self.word2idx.get(token, self.word2idx[self.SPECIAL_TOKENS['unk']])
            for token in tokens
        ]
        
        if add_special_tokens:
            start_idx = self.word2idx[self.SPECIAL_TOKENS['start']]
            end_idx = self.word2idx[self.SPECIAL_TOKENS['end']]
            indices = [start_idx] + indices + [end_idx]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Convert indices back to sentence."""
        tokens = []
        special_tokens = set(self.SPECIAL_TOKENS.values())
        
        for idx in indices:
            token = self.idx2word.get(idx, self.SPECIAL_TOKENS['unk'])
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        
        return " ".join(tokens)
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'word_counts': dict(self.word_counts),
                'min_freq': self.min_freq
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab = cls(min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_counts = Counter(data['word_counts'])
        return vocab