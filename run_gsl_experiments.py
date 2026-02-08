"""
Main script for running GSL gloss-to-text translation experiments.

Usage:
    python run_gsl_experiments.py --grid quick
    python run_gsl_experiments.py --grid focused
    python run_gsl_experiments.py --grid full
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.gsl_loader import (
    load_gsl_phoenix_manifest_df,
    preprocess_gsl_df,
    make_splits_from_split_column,
    save_splits,
    load_saved_splits,
)
from utils import (
    build_word_vocab,
    TranslationDataset,
    collate_batch,
)
from config.experiment_config import (
    create_default_grid,
    create_quick_grid,
    create_focused_grid,
    ExperimentConfig,
)
from experiments.runner import ExperimentRunner


def setup_data(data_dir: str, input_root: str, refresh: bool = False):
    """Load and prepare GSL dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if refresh or not (data_dir / "train.parquet").exists():
        print("Loading GSL Phoenix manifests...")
        raw_df = load_gsl_phoenix_manifest_df(input_root)
        print(f"Loaded {len(raw_df)} examples")
        
        print("Preprocessing...")
        df = preprocess_gsl_df(raw_df)
        print(f"After preprocessing: {len(df)} examples")
        
        splits = make_splits_from_split_column(df)
        print(f"Split sizes - Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")
        
        save_splits(splits, str(data_dir))
    else:
        print("Loading cached splits...")
        splits = load_saved_splits(str(data_dir))
        print(f"Split sizes - Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")
    
    return splits


def create_dataloaders(splits, batch_size: int = 32, num_workers: int = 0):
    """Create vocabularies and dataloaders."""
    print("Building vocabularies...")
    text_vocab = build_word_vocab(
        splits.train["processed_text"],
        specials=["<pad>", "<unk>"]
    )
    gloss_vocab = build_word_vocab(
        splits.train["processed_gloss"],
        specials=["<pad>", "<unk>", "<start>", "<end>"]
    )
    
    print(f"Text vocab size: {len(text_vocab.tokens)}")
    print(f"Gloss vocab size: {len(gloss_vocab.tokens)}")
    
    train_ds = TranslationDataset(splits.train, text_vocab, gloss_vocab)
    val_ds = TranslationDataset(splits.val, text_vocab, gloss_vocab)
    test_ds = TranslationDataset(splits.test, text_vocab, gloss_vocab)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
    )
    
    return train_loader, val_loader, test_loader, text_vocab, gloss_vocab


def main():
    parser = argparse.ArgumentParser(description="Run GSL gloss-to-text experiments")
    parser.add_argument(
        "--grid",
        type=str,
        default="quick",
        choices=["quick", "focused", "full"],
        help="Which parameter grid to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/gsl_phoenix",
        help="Directory to cache processed data"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="./data",
        help="Root directory of GSL Phoenix manifests"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory for experiment outputs"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="runs/results.csv",
        help="Path to results CSV file"
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Reprocess data from scratch"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip experiments that already have results"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 80)
    print("GSL Gloss-to-Text Translation Experiments")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    splits = setup_data(args.data_dir, args.input_root, args.refresh_data)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, text_vocab, gloss_vocab = create_dataloaders(
        splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create experiment grid
    print(f"\nCreating {args.grid} parameter grid...")
    if args.grid == "quick":
        grid = create_quick_grid()
    elif args.grid == "focused":
        grid = create_focused_grid()
    else:
        grid = create_default_grid()
    
    configs = grid.generate_configs()
    print(f"Generated {len(configs)} experiment configurations")
    
    # Preview first few configs
    print("\nFirst 3 configurations:")
    for i, config in enumerate(configs[:3], 1):
        print(f"  {i}. {config.get_run_name()}")
    if len(configs) > 3:
        print(f"  ... and {len(configs) - 3} more")
    
    # Run experiments
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("Starting experiments...")
    print("=" * 80)
    
    checkpoints = runner.run_experiments(
        configs=configs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        text_vocab=text_vocab,
        gloss_vocab=gloss_vocab,
        skip_if_exists=args.skip_existing,
    )
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    print(f"Results saved to: {args.results_csv}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Total experiments completed: {len(checkpoints)}")


if __name__ == "__main__":
    main()