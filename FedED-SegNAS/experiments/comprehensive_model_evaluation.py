#!/usr/bin/env python3
"""
Comprehensive Model Evaluation: All 8 Disease Models
====================================================

This script trains and evaluates Fuzzy CNN on all 8 disease models to provide
comprehensive performance metrics across different epistasis patterns.

Models Tested:
- model1: 2-way epistasis with marginal effects (Heritability 0.10)
- model2: 2-way epistasis with marginal effects (Heritability 0.10)
- model3: 2-way epistasis with marginal effects (Heritability 0.10)
- model4: 3-way epistasis (mixed, Heritability 0.10)
- model5: 2-way PURE epistasis - NO marginal effects (Heritability 0.10)
- model6: 2-way PURE epistasis - NO marginal effects (Heritability 0.10)
- model7: 2-way epistasis with marginal effects (Heritability 0.10)
- model8: 2-way epistasis with marginal effects (Heritability 0.10)

Expected Performance:
- Models with marginal effects (1,2,3,7,8): 85-95% accuracy
- Pure epistasis models (5,6): 70-85% accuracy
- Model 4 (3-way): 75-88% accuracy

Usage:
------
# Default: 30 epochs, 50 SNPs, test all available datasets
python experiments/comprehensive_model_evaluation.py

# Custom epochs
python experiments/comprehensive_model_evaluation.py --epochs 50

# Test specific SNP size
python experiments/comprehensive_model_evaluation.py --snps 100 --epochs 30

# Quick test (1 dataset per model)
python experiments/comprehensive_model_evaluation.py --quick --epochs 30

Author: FedED-SegNAS Project
Date: October 2024
"""

import sys
import os

# Add parent directory to path (works from any location)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.fuzzy_cnn import build_fuzzy_cnn
import time
import json
from datetime import datetime


# Model characteristics for reference
MODEL_INFO = {
    'model1': {'type': 'Marginal', 'order': 2, 'heritability': 0.10, 'expected_acc': '85-95%'},
    'model2': {'type': 'Marginal', 'order': 2, 'heritability': 0.10, 'expected_acc': '85-95%'},
    'model3': {'type': 'Marginal', 'order': 2, 'heritability': 0.10, 'expected_acc': '85-95%'},
    'model4': {'type': 'Mixed (3-way)', 'order': 3, 'heritability': 0.10, 'expected_acc': '75-88%'},
    'model5': {'type': 'Pure Epistasis', 'order': 2, 'heritability': 0.10, 'expected_acc': '70-85%'},
    'model6': {'type': 'Pure Epistasis', 'order': 2, 'heritability': 0.10, 'expected_acc': '70-85%'},
    'model7': {'type': 'Marginal', 'order': 2, 'heritability': 0.10, 'expected_acc': '85-95%'},
    'model8': {'type': 'Marginal', 'order': 2, 'heritability': 0.10, 'expected_acc': '85-95%'},
}


def find_datasets(model_name, snp_size=None):
    """
    Find all available datasets for a model.
    
    Returns list of tuples: (order, num_snps, dataset_id, filepath)
    """
    datasets = []
    model_dir = f'data/processed/{model_name}'
    
    if not os.path.exists(model_dir):
        return []
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                parts = full_path.split('/')
                
                # Extract order
                order = 2 if 'order2' in full_path else 3
                
                # Extract num_snps
                num_snps = None
                for part in parts:
                    if part.startswith('snps'):
                        num_snps = int(part.replace('snps', ''))
                        break
                
                if num_snps is None:
                    continue
                
                # Filter by SNP size if specified
                if snp_size and num_snps != snp_size:
                    continue
                
                # Extract dataset ID
                dataset_id = int(file.replace('dataset_', '').replace('.npz', ''))
                
                datasets.append((order, num_snps, dataset_id, full_path))
    
    return sorted(datasets)


def train_single_dataset(model_name, filepath, epochs=30, verbose=True):
    """
    Train Fuzzy CNN on a single dataset and return comprehensive metrics.
    """
    if verbose:
        print(f"\n  📊 Loading data from: {os.path.basename(filepath)}")
    
    try:
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Combine client data for centralized baseline
        X_train_list = []
        y_train_list = []
        
        metadata = data['metadata'][0]
        num_clients = metadata['num_clients']
        
        for i in range(num_clients):
            X_train_list.append(data[f'client_{i}_X'])
            y_train_list.append(data[f'client_{i}_y'])
        
        X_train = np.vstack(X_train_list).astype(np.float32)
        y_train = np.concatenate(y_train_list).astype(np.int32)
        
        X_val = data['validation_X'].astype(np.float32)
        y_val = data['validation_y'].astype(np.int32)
        X_test = data['test_X'].astype(np.float32)
        y_test = data['test_y'].astype(np.int32)
        
        num_snps = X_train.shape[1]
        
        if verbose:
            print(f"  📈 Data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}, SNPs={num_snps}")
        
        # Build model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        if verbose:
            print(f"  🔧 Model: {model.count_params():,} parameters")
            print(f"  🏃 Training for {epochs} epochs...")
        
        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            verbose=1 if verbose else 0
        )
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        
        # Collect metrics
        result = {
            'model_name': model_name,
            'model_type': MODEL_INFO[model_name]['type'],
            'epistasis_order': MODEL_INFO[model_name]['order'],
            'num_snps': num_snps,
            'epochs': epochs,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'initial_train_loss': float(history.history['loss'][0]),
            'final_train_loss': float(history.history['loss'][-1]),
            'initial_train_acc': float(history.history['accuracy'][0]),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'initial_val_loss': float(history.history['val_loss'][0]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'initial_val_acc': float(history.history['val_accuracy'][0]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'training_time_seconds': float(training_time),
            'training_time_minutes': float(training_time / 60),
            'convergence': 'Yes' if test_accuracy > 0.6 else 'Needs More Epochs',
            'performance_category': 'Excellent' if test_accuracy >= 0.85 else ('Good' if test_accuracy >= 0.70 else 'Fair'),
            'expected_accuracy': MODEL_INFO[model_name]['expected_acc'],
            'status': 'SUCCESS',
            'filepath': filepath
        }
        
        if verbose:
            print(f"  ✅ Complete!")
            print(f"     Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"     Test Loss: {test_loss:.4f}")
            print(f"     Training Time: {training_time/60:.2f} minutes")
            print(f"     Performance: {result['performance_category']}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Error: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e),
            'filepath': filepath
        }


def run_comprehensive_evaluation(epochs=30, snp_size=50, quick=False):
    """
    Run comprehensive evaluation on all 8 models.
    """
    print("\n" + "="*80)
    print("🧬 COMPREHENSIVE FUZZY CNN EVALUATION: ALL 8 DISEASE MODELS")
    print("="*80)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - SNP Size: {snp_size}")
    print(f"   - Quick Mode: {quick}")
    print("="*80)
    
    all_results = []
    total_start = time.time()
    
    for model_num in range(1, 9):
        model_name = f"model{model_num}"
        
        print(f"\n{'='*80}")
        print(f"🔬 TESTING {model_name.upper()}")
        print(f"{'='*80}")
        print(f"Type: {MODEL_INFO[model_name]['type']}")
        print(f"Order: {MODEL_INFO[model_name]['order']}-way epistasis")
        print(f"Expected Accuracy: {MODEL_INFO[model_name]['expected_acc']}")
        print("-"*80)
        
        # Find datasets
        datasets = find_datasets(model_name, snp_size)
        
        if not datasets:
            print(f"⚠️  No datasets with {snp_size} SNPs found for {model_name}")
            print(f"   Searching for any available datasets...")
            datasets = find_datasets(model_name)
            if datasets:
                available_snps = sorted(set([d[1] for d in datasets]))
                print(f"   Available SNP sizes: {available_snps}")
            else:
                print(f"   No datasets found at all for {model_name}")
            continue
        
        print(f"📁 Found {len(datasets)} dataset(s) with {snp_size} SNPs")
        
        # In quick mode, test only first dataset
        if quick:
            datasets = datasets[:1]
            print(f"⚡ Quick mode: Testing only first dataset")
        
        # Train on each dataset
        for idx, (order, num_snps, dataset_id, filepath) in enumerate(datasets, 1):
            print(f"\n{'─'*80}")
            print(f"📊 Dataset {idx}/{len(datasets)}: order{order}/snps{num_snps}/dataset_{dataset_id}")
            print(f"{'─'*80}")
            
            result = train_single_dataset(model_name, filepath, epochs, verbose=True)
            
            if result['status'] == 'SUCCESS':
                all_results.append(result)
    
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print(f"✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"⏱️  Total Time: {total_time/60:.2f} minutes")
    print(f"📊 Total Successful Runs: {len(all_results)}")
    print("="*80 + "\n")
    
    return all_results


def save_results(results, output_dir='results/comprehensive_evaluation'):
    """Save results to multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = f'{output_dir}/results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 Results saved to: {csv_path}")
    
    # Save to JSON (with full details)
    json_path = f'{output_dir}/results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {json_path}")
    
    # Save summary report
    summary_path = f'{output_dir}/summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE FUZZY CNN EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models Tested: {df['model_name'].nunique()}\n")
        f.write(f"Total Datasets: {len(results)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Test Accuracy: {df['test_accuracy'].mean():.4f} ({df['test_accuracy'].mean()*100:.2f}%)\n")
        f.write(f"Std Dev: {df['test_accuracy'].std():.4f}\n")
        f.write(f"Min: {df['test_accuracy'].min():.4f}\n")
        f.write(f"Max: {df['test_accuracy'].max():.4f}\n")
        f.write(f"Median: {df['test_accuracy'].median():.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("RESULTS BY MODEL\n")
        f.write("-"*80 + "\n")
        summary = df.groupby('model_name').agg({
            'test_accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'training_time_minutes': 'mean'
        })
        f.write(str(summary))
        f.write("\n\n")
        
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE CATEGORIES\n")
        f.write("-"*80 + "\n")
        perf_counts = df['performance_category'].value_counts()
        for category, count in perf_counts.items():
            f.write(f"{category}: {count} ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"💾 Summary saved to: {summary_path}")
    
    return df


def create_visualizations(df, output_dir='results/comprehensive_evaluation'):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Main accuracy comparison
    plt.figure(figsize=(16, 8))
    
    summary = df.groupby('model_name').agg({
        'test_accuracy': ['mean', 'std']
    })
    
    means = summary['test_accuracy']['mean'].values
    stds = summary['test_accuracy']['std'].fillna(0).values
    models = summary.index.tolist()
    
    colors = ['green' if m >= 0.85 else 'orange' if m >= 0.70 else 'red' for m in means]
    
    bars = plt.bar(range(len(models)), means, yerr=stds, capsize=5,
                   color=colors, edgecolor='black', linewidth=2, alpha=0.75)
    
    plt.xticks(range(len(models)), models, fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title('Fuzzy CNN Performance: All 8 Disease Models', fontsize=16, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            label = f'{m:.3f}\n±{s:.3f}'
        else:
            label = f'{m:.3f}'
        plt.text(i, m + s + 0.02, label, ha='center', fontsize=10, fontweight='bold')
    
    # Add expected ranges
    for i, model in enumerate(models):
        expected = MODEL_INFO[model]['expected_acc']
        plt.text(i, 0.42, f'Expected:\n{expected}', ha='center', fontsize=8, style='italic')
    
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='green', label='Excellent (≥85%)'),
        Patch(facecolor='orange', label='Good (70-85%)'),
        Patch(facecolor='red', label='Fair (<70%)')
    ]
    plt.legend(handles=legend, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: accuracy_comparison_{timestamp}.png")
    
    # 2. Training time comparison
    plt.figure(figsize=(14, 6))
    time_summary = df.groupby('model_name')['training_time_minutes'].mean().sort_values()
    plt.barh(range(len(time_summary)), time_summary.values, color='steelblue', edgecolor='black', linewidth=1.5)
    plt.yticks(range(len(time_summary)), time_summary.index, fontsize=11)
    plt.xlabel('Average Training Time (minutes)', fontsize=12, fontweight='bold')
    plt.title('Training Time by Model', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(time_summary.values):
        plt.text(v + 0.1, i, f'{v:.2f}m', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: training_time_{timestamp}.png")
    
    # 3. Learning curves (if multiple epochs)
    if df['epochs'].iloc[0] >= 10:
        print(f"📊 Note: Learning curves not generated (requires history data)")
    
    print(f"✅ All visualizations saved to: {output_dir}/")


def print_summary_table(df):
    """Print a nice summary table."""
    print("\n" + "="*80)
    print("📊 SUMMARY TABLE")
    print("="*80)
    
    summary = df.groupby('model_name').agg({
        'test_accuracy': ['mean', 'std', 'min', 'max'],
        'model_type': 'first',
        'performance_category': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
    }).round(4)
    
    print(summary.to_string())
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of Fuzzy CNN on all 8 disease models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation (30 epochs, 50 SNPs)
  python experiments/comprehensive_model_evaluation.py
  
  # Custom epochs
  python experiments/comprehensive_model_evaluation.py --epochs 50
  
  # Different SNP size
  python experiments/comprehensive_model_evaluation.py --snps 100 --epochs 30
  
  # Quick test (1 dataset per model)
  python experiments/comprehensive_model_evaluation.py --quick --epochs 10
        """
    )
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--snps', type=int, default=50,
                       help='SNP size to test (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: test only 1 dataset per model')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧬 FUZZY CNN COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"⚙️  Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   SNP Size: {args.snps}")
    print(f"   Quick Mode: {args.quick}")
    print("="*80 + "\n")
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        epochs=args.epochs,
        snp_size=args.snps,
        quick=args.quick
    )
    
    if not results:
        print("\n❌ No results collected! Check dataset availability.")
        return 1
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print_summary_table(df)
    
    # Save results
    df_saved = save_results(results)
    
    # Create visualizations
    create_visualizations(df_saved)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE EVALUATION COMPLETE")
    print("="*80)
    print(f"Models Tested: {df['model_name'].nunique()}/8")
    print(f"Total Datasets: {len(results)}")
    print(f"Average Accuracy: {df['test_accuracy'].mean():.4f} ({df['test_accuracy'].mean()*100:.2f}%)")
    print(f"Results Directory: results/comprehensive_evaluation/")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
