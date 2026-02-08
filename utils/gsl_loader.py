"""Data loading and preprocessing for GSL Phoenix dataset."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import re
import string


_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def remove_noise(txt: str) -> str:
    if txt is None:
        return ""
    txt = re.sub(r"\d+", "", str(txt))
    txt = txt.translate(_PUNCT_TABLE)
    txt = " ".join(txt.split())
    return txt


def preprocess_text(text: str) -> str:
    return remove_noise(str(text).lower())


def preprocess_gloss(gloss: str) -> str:
    return remove_noise(str(gloss).upper())


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_gsl_phoenix_manifest_df(input_root: str = "../input/gsl-glosses-and-text") -> pd.DataFrame:
    """Load GSL Phoenix manifests and return DataFrame with text/gloss columns."""
    input_root = Path(input_root)

    manifest_map = {
        "train": input_root / "train_rgb_manifest.json",
        "dev":   input_root / "dev_rgb_manifest.json",
        "test":  input_root / "test_rgb_manifest.json",
    }

    rows = []
    for split_name, p in manifest_map.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data:
            if "success" in ex and not ex["success"]:
                continue

            orth = ex.get("orth", "")
            translation = ex.get("translation", "")

            rows.append({
                "video_id": ex.get("video_id", ""),
                "split": ex.get("split", split_name),
                "text": translation,
                "gloss": orth,
            })

    df = pd.DataFrame(rows)

    if "text" not in df.columns or "gloss" not in df.columns:
        raise ValueError("Expected to build columns: text, gloss")

    df["split"] = df["split"].astype(str).str.lower().replace({"valid": "dev", "val": "dev"})

    return df


def preprocess_gsl_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to GSL dataframe."""
    df = df.copy()
    df["processed_text"] = df["text"].apply(preprocess_text)
    df["processed_gloss"] = df["gloss"].apply(preprocess_gloss)
    df = df[(df["processed_text"] != "") & (df["processed_gloss"] != "")]
    df = df.reset_index(drop=True)
    return df


def make_splits_from_split_column(df: pd.DataFrame) -> DatasetSplits:
    """Create DatasetSplits using existing split column."""
    df = df.copy()
    if "split" not in df.columns:
        raise ValueError("df must contain a 'split' column")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"].isin(["dev", "valid", "val"])].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Bad split sizes: train={len(train_df)} dev={len(val_df)} test={len(test_df)}. "
            f"Unique splits found: {sorted(df['split'].unique().tolist())}"
        )

    return DatasetSplits(train=train_df, val=val_df, test=test_df)


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
