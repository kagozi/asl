from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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
