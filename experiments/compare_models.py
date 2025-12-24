# ============================================================================
# experiments/compare_models.py
# ============================================================================
"""Compare baseline and modern transformer models."""

import pandas as pd
from pathlib import Path
import argparse
import json
from utils import plot_metric_comparison, save_metrics_table


def main():
    parser = argparse.ArgumentParser(description='Compare model results')
    parser.add_argument('--results-dir', type=str, default='../santosh_lab/shared/KagoziA/sl/results',
                        help='Directory containing result folders')
    parser.add_argument('--output', type=str, default='../santosh_lab/shared/KagoziA/sl/results/comparison',
                        help='Output directory for comparison plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all results
    all_results = {}
    
    for result_folder in results_dir.iterdir():
        if not result_folder.is_dir():
            continue
        
        predictions_file = result_folder / 'predictions.json'
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                data = json.load(f)
                model_name = result_folder.name
                all_results[model_name] = data['metrics']
    
    if not all_results:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Found results for {len(all_results)} models:")
    for model_name in all_results.keys():
        print(f"  - {model_name}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    
    df = pd.DataFrame(all_results).T
    df = df.round(2)
    print(df.to_string())
    
    # Save comparison table
    save_metrics_table(all_results, output_dir / 'comparison.csv')
    save_metrics_table(all_results, output_dir / 'comparison.md')
    
    # Plot comparison
    plot_metric_comparison(
        all_results,
        output_dir / 'comparison.png',
        title="Baseline vs Modern Transformer Comparison"
    )
    
    # Analyze improvements
    print("\n" + "="*80)
    print("IMPROVEMENTS (Modern vs Baseline)")
    print("="*80 + "\n")
    
    for direction in ['text2gloss', 'gloss2text']:
        baseline_key = [k for k in all_results.keys() if f'baseline_{direction}' in k]
        modern_key = [k for k in all_results.keys() if f'modern_{direction}' in k]
        
        if baseline_key and modern_key:
            baseline_key = baseline_key[0]
            modern_key = modern_key[0]
            
            print(f"\n{direction.upper()}:")
            baseline_metrics = all_results[baseline_key]
            modern_metrics = all_results[modern_key]
            
            for metric in baseline_metrics.keys():
                baseline_val = baseline_metrics[metric]
                modern_val = modern_metrics[metric]
                improvement = modern_val - baseline_val
                percent = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                
                print(f"  {metric:15s}: {baseline_val:6.2f} → {modern_val:6.2f} "
                      f"(+{improvement:5.2f}, +{percent:5.1f}%)")
    
    print(f"\nComparison saved to {output_dir}")


if __name__ == '__main__':
    main()