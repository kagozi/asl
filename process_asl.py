import torch
from dataclasses import dataclass
from typing import Tuple
from typing import List, Tuple
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_aslg_pc12_df() -> pd.DataFrame:
    """Load ASLG-PC12 from HuggingFace.

    Expects columns: 'text' and 'gloss'.
    """
    ds = load_dataset("achrafothman/aslg_pc12", split="train")
    df = ds.to_pandas()
    if "text" not in df.columns or "gloss" not in df.columns:
        raise ValueError("Expected columns: text, gloss")
    return df


def preprocess_aslg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["processed_text"] = df["text"].apply(preprocess_text)
    df["processed_gloss"] = df["gloss"].apply(preprocess_gloss)
    # Drop empties
    df = df[(df["processed_text"] != "") & (df["processed_gloss"] != "")]
    df = df.reset_index(drop=True)
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: int = 82710,
    val_size: int = 4000,
    seed: int = 42,
) -> DatasetSplits:
    """Match the ASLG-PC12 split sizes used in your draft."""
    if len(df) < train_size + val_size + 1:
        raise ValueError(f"Dataset too small for requested split: {len(df)}")

    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=seed, shuffle=True)

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def save_splits(splits: DatasetSplits, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    splits.train.to_parquet(f"{out_dir}/train.parquet", index=False)
    splits.val.to_parquet(f"{out_dir}/val.parquet", index=False)
    splits.test.to_parquet(f"{out_dir}/test.parquet", index=False)


def load_saved_splits(out_dir: str) -> DatasetSplits:
    train = pd.read_parquet(f"{out_dir}/train.parquet")
    val = pd.read_parquet(f"{out_dir}/val.parquet")
    test = pd.read_parquet(f"{out_dir}/test.parquet")
    return DatasetSplits(train=train, val=val, test=test)



class TranslationDataset(Dataset):
    def __init__(self, df, text_vocab: Vocab, gloss_vocab: Vocab):
        self.df = df
        self.text_vocab = text_vocab
        self.gloss_vocab = gloss_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        src_ids = encode(row["processed_text"], self.text_vocab, add_bos_eos=False)
        tgt_ids = encode(row["processed_gloss"], self.gloss_vocab, add_bos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


@dataclass
class Batch:
    src: torch.Tensor
    tgt: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], text_vocab: Vocab, gloss_vocab: Vocab) -> Batch:
    src_list = [b[0] for b in batch]
    tgt_list = [b[1] for b in batch]

    src = pad_sequence(src_list, batch_first=True, padding_value=text_vocab.pad_id)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=gloss_vocab.pad_id)

    src_key_padding_mask = src.eq(text_vocab.pad_id)  # (B, S)
    tgt_key_padding_mask = tgt.eq(gloss_vocab.pad_id)  # (B, T)

    return Batch(src=src, tgt=tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)