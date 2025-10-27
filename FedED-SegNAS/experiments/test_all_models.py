#!/usr/bin/env python3
"""
Comprehensive Model Testing: ALL 8 Disease Models
=================================================

Tests Fuzzy CNN on all 8 disease models to verify performance across:
- model1: 2-way epistasis with marginal effects
- model2: 2-way epistasis with marginal effects  
- model3: 2-way epistasis with marginal effects
- model4: 3-way epistasis (mixed)
- model5: 2-way pure epistasis (no marginal)
- model6: 2-way pure epistasis (no marginal)
- model7: 2-way epistasis with marginal effects
- model8: 2-way epistasis with marginal effects

Usage:
------
python experiments/test_all_models.py --epochs 50
python experiments/test_all_models.py --epochs 10 --quick
"""

import sys
sys.path.insert(0, '/app/FedED-SegNAS')

import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.fuzzy_cnn import build_fuzzy_cnn
import time
import json


def get_available_datasets(model_name):
    """Find all available datasets for a model."""
    datasets = []
    model_dir = f'data/processed/{model_name}'
    
    if not os.path.exists(model_dir):
        return []
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                # Extract order and num_snps from path
                parts = full_path.split('/')
                if 'order2' in full_path or 'order3' in full_path:
                    order = 2 if 'order2' in full_path else 3
                    for part in parts:
                        if part.startswith('snps'):
                            num_snps = int(part.replace('snps', ''))
                            dataset_id = int(file.replace('dataset_', '').replace('.npz', ''))
                            datasets.append((order, num_snps, dataset_id, full_path))
    
    return datasets


def train_and_evaluate_single(model_name, filepath, epochs=20):
    """Train and evaluate on a single dataset."""
    
    try:
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Combine client data
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
        
        # Build model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        
        return {
            'model_name': model_name,
            'num_snps': num_snps,
            'epochs': epochs,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'training_time': float(training_time),
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAILED',
            'error': str(e)
        }


def test_all_models(epochs=20, quick=False, snp_size=50):
    """Test Fuzzy CNN on all 8 models."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FUZZY CNN TESTING: ALL 8 MODELS")
    print("="*70)
    print(f"Epochs: {epochs}")
    print(f"SNP Size: {snp_size}")
    print(f"Quick Mode: {quick}")
    print("="*70 + "\n")
    
    all_results = []
    
    for model_num in range(1, 9):
        model_name = f"model{model_num}"
        
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*70}")
        
        # Get datasets
        datasets = get_available_datasets(model_name)
        
        if not datasets:
            print(f"⚠️ No datasets found for {model_name}")
            continue
        
        # Filter by SNP size
        datasets = [d for d in datasets if d[1] == snp_size]
        
        if not datasets:
            print(f"⚠️ No datasets with {snp_size} SNPs found for {model_name}")
            continue
        
        print(f"Found {len(datasets)} datasets with {snp_size} SNPs")
        
        # In quick mode, test only first dataset
        if quick:
            datasets = datasets[:1]
        
        for order, num_snps, dataset_id, filepath in datasets:
            print(f"\n  Training on: order{order}/snps{num_snps}/dataset_{dataset_id}")
            print(f"  File: {filepath}")
            
            result = train_and_evaluate_single(model_name, filepath, epochs)
            
            if result['status'] == 'SUCCESS':
                print(f"  ✅ Test Acc: {result['test_accuracy']:.4f} | "
                      f"Time: {result['training_time']:.1f}s")
                all_results.append(result)
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return all_results


def save_and_visualize_results(results, output_dir='results/all_models_test'):
    """Save results and create visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("\n❌ No results to save!")
        return
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = f'{output_dir}/all_models_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save to JSON
    json_path = f'{output_dir}/all_models_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {json_path}")
    
    # Create summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    summary = df.groupby('model_name')['test_accuracy'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(summary)
    
    # Overall statistics
    print(f"\nOverall Test Accuracy:")
    print(f"  Mean: {df['test_accuracy'].mean():.4f}")
    print(f"  Std:  {df['test_accuracy'].std():.4f}")
    print(f"  Min:  {df['test_accuracy'].min():.4f}")
    print(f"  Max:  {df['test_accuracy'].max():.4f}")
    
    # Create visualization
    plt.figure(figsize=(14, 6))
    
    # Bar plot with error bars
    summary_mean = df.groupby('model_name')['test_accuracy'].mean().sort_index()
    summary_std = df.groupby('model_name')['test_accuracy'].std().fillna(0)
    
    colors = []
    for acc in summary_mean.values:
        if acc >= 0.85:
            colors.append('green')
        elif acc >= 0.70:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = plt.bar(range(len(summary_mean)), summary_mean.values, 
                   yerr=summary_std.values, capsize=5,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    plt.xticks(range(len(summary_mean)), summary_mean.index, rotation=0)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Fuzzy CNN Performance Across All 8 Disease Models', 
              fontsize=14, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (v, err) in enumerate(zip(summary_mean.values, summary_std.values)):
        if err > 0:
            label = f'{v:.3f}±{err:.3f}'
        else:
            label = f'{v:.3f}'
        plt.text(i, v + err + 0.02, label, ha='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Excellent (≥85%)'),
        Patch(facecolor='orange', label='Good (70-85%)'),
        Patch(facecolor='red', label='Needs Improvement (<70%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_models_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_dir}/all_models_accuracy.png")
    
    # Create detailed comparison table
    comparison_table = df.pivot_table(
        values='test_accuracy',
        index='model_name',
        aggfunc=['mean', 'std', 'count']
    )
    
    print("\n" + "="*70)
    print("DETAILED COMPARISON TABLE")
    print("="*70)
    print(comparison_table)
    print("="*70)
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test Fuzzy CNN on all 8 disease models'
    )
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: test only 1 dataset per model')
    parser.add_argument('--snps', type=int, default=50,
                       help='SNP size to test (default: 50)')
    
    args = parser.parse_args()
    
    # Run tests
    results = test_all_models(
        epochs=args.epochs,
        quick=args.quick,
        snp_size=args.snps
    )
    
    if not results:
        print("\n❌ No results collected!")
        return
    
    # Save and visualize
    df = save_and_visualize_results(results)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE TESTING COMPLETE")
    print("="*70)
    print(f"Total models tested: 8")
    print(f"Total datasets tested: {len(results)}")
    print(f"Average accuracy: {df['test_accuracy'].mean():.4f}")
    print(f"Results saved to: results/all_models_test/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
