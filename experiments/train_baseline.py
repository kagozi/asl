# ============================================================================
# experiments/train_baseline.py
# ============================================================================
"""Training script for baseline transformer model."""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import load_config, save_config
from data.dataset import load_and_prepare_data, collate_fn
from models.baseline_transformer import create_baseline_model
from training.trainer import TransformerTrainer
from evaluation.evaluator import TranslationEvaluator
from utils.visualization import plot_training_curves, save_metrics_table


def main():
    parser = argparse.ArgumentParser(description='Train baseline transformer model')
    parser.add_argument('--config', type=str, default='config/baseline_config.yaml',
                        help='Path to config file')
    parser.add_argument('--direction', type=str, default='text2gloss',
                        choices=['text2gloss', 'gloss2text', 'both'],
                        help='Translation direction')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config['compute']['device'] = str(device)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Train for specified direction(s)
    directions = ['text2gloss', 'gloss2text'] if args.direction == 'both' else [args.direction]
    
    for direction in directions:
        print("\n" + "="*80)
        print(f"Training Baseline Transformer: {direction.upper()}")
        print("="*80 + "\n")
        
        train_single_direction(config, direction, device, args.resume)


def train_single_direction(config, direction, device, resume_path=None):
    """Train model for a single direction."""
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / f'baseline_{direction}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    save_config(config, output_dir / 'config.yaml')
    
    # Load and prepare data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_and_prepare_data(
        config, direction
    )
    
    # Save vocabularies
    src_vocab.save(output_dir / 'src_vocab.json')
    tgt_vocab.save(output_dir / 'tgt_vocab.json')
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['compute']['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0),
        num_workers=num_workers,
        pin_memory=config['compute']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0),
        num_workers=num_workers,
        pin_memory=config['compute']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0),
        num_workers=num_workers,
        pin_memory=config['compute']['pin_memory']
    )
    
    # Create model
    print("Creating model...")
    model = create_baseline_model(config, len(src_vocab), len(tgt_vocab))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(Path(resume_path))
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Plot training curves
    metrics = trainer.get_metrics()
    plot_training_curves(
        metrics['train_losses'],
        metrics['val_losses'],
        metrics['learning_rates'],
        output_dir / 'training_curves.png'
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    # Load best model
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        print(f"Loading best model from {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = TranslationEvaluator(
        model=model,
        device=device,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_gen_len=config['evaluation']['max_generation_length'],
        beam_size=config['evaluation']['beam_size']
    )
    
    # Evaluate
    test_metrics = evaluator.evaluate(
        test_loader,
        num_examples=config['evaluation'].get('num_eval_examples'),
        save_predictions=True,
        output_path=output_dir / 'predictions.json'
    )
    
    # Print metrics
    evaluator.print_metrics(test_metrics)
    
    # Show examples
    evaluator.show_examples(test_loader, num_examples=5)
    
    # Save results to CSV
    results = {
        f'baseline_{direction}': test_metrics
    }
    
    # Append to global results file
    results_file = Path('results/scores.csv')
    save_metrics_table(results, results_file)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Checkpoints saved to {checkpoint_dir}")


if __name__ == '__main__':
    main()