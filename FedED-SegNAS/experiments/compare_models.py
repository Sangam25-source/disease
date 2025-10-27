#!/usr/bin/env python3
"""
Model Comparison Script for Fuzzy CNN
======================================

Trains Fuzzy CNN on multiple disease models and compares performance.

This script helps understand how well the Fuzzy CNN performs on:
- Pure epistasis models (model1, model5) - harder to detect
- Marginal effect models (model3, model7) - easier to detect

Usage:
------
python experiments/compare_models.py
python experiments/compare_models.py --epochs 50 --quick

Author: FedED-SegNAS Project
Date: October 2024
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
import json
import time


def train_and_evaluate(model_name, order, num_snps, dataset_id, epochs=50):
    """
    Train and evaluate model on a single dataset configuration.
    
    Returns:
    --------
    dict : Training results
    """
    print(f"\n{'='*70}")
    print(f"Training: {model_name} | order{order} | SNPs={num_snps} | dataset={dataset_id}")
    print(f"{'='*70}")
    
    # Load data
    filepath = f'data/processed/{model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}.npz'
    
    if not os.path.exists(filepath):
        print(f"⚠️ Dataset not found: {filepath}")
        return None
    
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
    
    print(f"Data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}, SNPs={num_snps}")
    
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
    
    print(f"✅ Complete: Test Acc={test_accuracy:.4f} | Time={training_time:.1f}s")
    
    return {
        'model_name': model_name,
        'order': order,
        'num_snps': num_snps,
        'dataset_id': dataset_id,
        'epochs': epochs,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'training_time': float(training_time)
    }


def run_comparison(models_to_compare, epochs=50, quick=False):
    """
    Run comparison across multiple models.
    
    Parameters:
    -----------
    models_to_compare : list
        List of (model_name, order, num_snps) tuples
    epochs : int
        Number of training epochs
    quick : bool
        If True, only test one dataset per configuration
    """
    print("\n" + "="*70)
    print("FUZZY CNN MODEL COMPARISON")
    print("="*70)
    print(f"Models to compare: {len(models_to_compare)}")
    print(f"Epochs per model: {epochs}")
    print(f"Quick mode: {quick}")
    print("="*70 + "\n")
    
    all_results = []
    
    num_datasets = 1 if quick else 2
    
    for model_name, order, num_snps in models_to_compare:
        for dataset_id in range(num_datasets):
            try:
                result = train_and_evaluate(
                    model_name=model_name,
                    order=order,
                    num_snps=num_snps,
                    dataset_id=dataset_id,
                    epochs=epochs
                )
                
                if result:
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Error training {model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}: {e}")
    
    return all_results


def save_results(results, output_dir='results/comparison'):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = f'{output_dir}/model_comparison_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save to JSON
    json_path = f'{output_dir}/model_comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {json_path}")
    
    return df


def plot_comparison(df, output_dir='results/comparison'):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar plot: Accuracy by model
    plt.figure(figsize=(12, 6))
    
    model_accuracy = df.groupby('model_name')['test_accuracy'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(model_accuracy)), model_accuracy.values, color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Color bars based on performance
    for i, (model, acc) in enumerate(model_accuracy.items()):
        if acc >= 0.85:
            bars[i].set_color('green')
        elif acc >= 0.70:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    plt.xticks(range(len(model_accuracy)), model_accuracy.index, rotation=0)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Fuzzy CNN Performance Across Disease Models', fontsize=14, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(model_accuracy.values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {output_dir}/accuracy_by_model.png")
    
    # 2. Heatmap: Accuracy by model and SNP size
    if 'num_snps' in df.columns and len(df['num_snps'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        
        pivot = df.pivot_table(
            values='test_accuracy',
            index='model_name',
            columns='num_snps',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0.5, vmax=1.0, cbar_kws={'label': 'Test Accuracy'})
        plt.title('Test Accuracy: Model vs SNP Count', fontsize=14, fontweight='bold')
        plt.xlabel('Number of SNPs', fontsize=12)
        plt.ylabel('Disease Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Plot saved: {output_dir}/accuracy_heatmap.png")
    
    # 3. Training time comparison
    plt.figure(figsize=(12, 6))
    
    model_time = df.groupby('model_name')['training_time'].mean().sort_values()
    
    plt.barh(range(len(model_time)), model_time.values, color='lightcoral', edgecolor='black', linewidth=1.5)
    plt.yticks(range(len(model_time)), model_time.index)
    plt.xlabel('Average Training Time (seconds)', fontsize=12)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {output_dir}/training_time.png")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\nAverage Test Accuracy by Model:")
    print(df.groupby('model_name')['test_accuracy'].agg(['mean', 'std', 'min', 'max']))
    
    print("\nAverage Training Time by Model (seconds):")
    print(df.groupby('model_name')['training_time'].mean())
    
    print("\nOverall Statistics:")
    print(df[['test_accuracy', 'test_loss', 'training_time']].describe())
    
    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare Fuzzy CNN performance across multiple disease models'
    )
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: test only 1 dataset per configuration')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['model1', 'model5'],
                       help='Models to compare (default: model1 model5)')
    parser.add_argument('--snps', type=int, nargs='+',
                       default=[50, 100],
                       help='SNP sizes to test (default: 50 100)')
    
    args = parser.parse_args()
    
    # Create comparison configurations
    models_to_compare = []
    for model in args.models:
        for snps in args.snps:
            models_to_compare.append((model, 2, snps))  # order=2
    
    print(f"\nConfigurations to test: {len(models_to_compare)}")
    for model, order, snps in models_to_compare:
        print(f"  - {model}/order{order}/snps{snps}")
    
    # Run comparison
    results = run_comparison(
        models_to_compare=models_to_compare,
        epochs=args.epochs,
        quick=args.quick
    )
    
    if not results:
        print("❌ No results collected!")
        return
    
    # Save results
    df = save_results(results)
    
    # Create visualizations
    plot_comparison(df)
    
    # Print summary
    print_summary(df)
    
    print("\n✅ Comparison complete!")
    print(f"Total models trained: {len(results)}")
    print(f"Results saved to: results/comparison/")


if __name__ == '__main__':
    main()
