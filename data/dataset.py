# ============================================================================
# data/dataset.py
# ============================================================================
"""PyTorch Dataset classes for text-gloss translation."""

from data.preprocessing import Vocabulary
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional
import pandas as pd


class TranslationDataset(Dataset):
    """Dataset for text-gloss translation."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        src_vocab: 'Vocabulary',
        tgt_vocab:  'Vocabulary',
        direction: str = 'text2gloss',
        max_length: Optional[int] = None
    ):
        """
        Args:
            dataframe: DataFrame with 'processed_text' and 'processed_gloss' columns
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            direction: 'text2gloss' or 'gloss2text'
            max_length: Maximum sequence length (None = no limit)
        """
        self.data = dataframe.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.direction = direction
        self.max_length = max_length
        
        # Set source and target columns based on direction
        if direction == 'text2gloss':
            self.src_col = 'processed_text'
            self.tgt_col = 'processed_gloss'
        elif direction == 'gloss2text':
            self.src_col = 'processed_gloss'
            self.tgt_col = 'processed_text'
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example."""
        src_text = self.data.iloc[idx][self.src_col]
        tgt_text = self.data.iloc[idx][self.tgt_col]
        
        # Encode sequences
        src_indices = self.src_vocab.encode(src_text, add_special_tokens=False)
        tgt_indices = self.tgt_vocab.encode(tgt_text, add_special_tokens=True)
        
        # Truncate if needed
        if self.max_length:
            src_indices = src_indices[:self.max_length]
            tgt_indices = tgt_indices[:self.max_length]
        
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long)
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], 
               src_pad_idx: int = 0, 
               tgt_pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (source, target) tensor pairs
        src_pad_idx: Padding index for source sequences
        tgt_pad_idx: Padding index for target sequences
    
    Returns:
        Padded source and target batches
    """
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]
    
    src_batch_padded = pad_sequence(
        src_batch, 
        batch_first=True, 
        padding_value=src_pad_idx
    )
    tgt_batch_padded = pad_sequence(
        tgt_batch, 
        batch_first=True, 
        padding_value=tgt_pad_idx
    )
    
    return src_batch_padded, tgt_batch_padded


def load_and_prepare_data(
    config: Dict,
    direction: str = 'text2gloss'
) -> Tuple[Dataset, Dataset, Dataset, 'Vocabulary', 'Vocabulary']:
    """
    Load and prepare datasets for training.
    
    Args:
        config: Configuration dictionary
        direction: 'text2gloss' or 'gloss2text'
    
    Returns:
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from .preprocessing import TextGlossPreprocessor, Vocabulary
    
    # Load dataset
    print(f"Loading dataset: {config['data']['dataset_name']}")
    dataset = load_dataset(config['data']['dataset_name'])
    df = pd.DataFrame(dataset['train'])
    
    # Preprocess
    print("Preprocessing data...")
    preprocessor = TextGlossPreprocessor(
        lowercase_text=config['data']['lowercase_text'],
        uppercase_gloss=config['data']['uppercase_gloss'],
        remove_punctuation=config['data']['remove_punctuation'],
        remove_digits=config['data']['remove_digits']
    )
    df = preprocessor.process_dataframe(df)
    
    # Split data
    print("Splitting data...")
    train_df, temp_df = train_test_split(
        df, 
        train_size=config['data']['train_size'], 
        random_state=config['seed']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=config['data']['val_size'], 
        random_state=config['seed']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Build vocabularies
    print("Building vocabularies...")
    if direction == 'text2gloss':
        src_vocab = Vocabulary()
        src_vocab.build_from_sentences(train_df['processed_text'].tolist())
        
        tgt_vocab = Vocabulary()
        tgt_vocab.build_from_sentences(train_df['processed_gloss'].tolist())
    else:  # gloss2text
        src_vocab = Vocabulary()
        src_vocab.build_from_sentences(train_df['processed_gloss'].tolist())
        
        tgt_vocab = Vocabulary()
        tgt_vocab.build_from_sentences(train_df['processed_text'].tolist())
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Create datasets
    max_length = config['data'].get('max_length', None)
    
    train_dataset = TranslationDataset(
        train_df, src_vocab, tgt_vocab, direction, max_length
    )
    val_dataset = TranslationDataset(
        val_df, src_vocab, tgt_vocab, direction, max_length
    )
    test_dataset = TranslationDataset(
        test_df, src_vocab, tgt_vocab, direction, max_length
    )
    
    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab