#!/usr/bin/env python3
"""
Comprehensive Model Evaluation - MODEL5 V2 (Optimized)
======================================================

Best approach from testing:
- ExtraTrees with n_estimators=800, max_depth=12, min_samples_split=2
- Achieves ~65% on order2 datasets

Usage:
------
python experiments/comprehensive_model_evaluation_model5_v2.py
"""

import sys
import os
import json
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')


def find_all_model5_datasets():
    """Find all available Model5 datasets."""
    datasets = []
    
    if os.path.exists('data/processed/model5'):
        base_path = 'data/processed/model5'
    elif os.path.exists('../data/processed/model5'):
        base_path = '../data/processed/model5'
    else:
        return []
    
    for order in ['order2', 'order3']:
        order_path = os.path.join(base_path, order)
        if not os.path.exists(order_path):
            continue
        
        for snp_folder in os.listdir(order_path):
            if snp_folder.startswith('snps'):
                snp_size = int(snp_folder.replace('snps', ''))
                snp_path = os.path.join(order_path, snp_folder)
                
                for file in os.listdir(snp_path):
                    if file.endswith('.npz'):
                        dataset_id = int(file.replace('dataset_', '').replace('.npz', ''))
                        full_path = os.path.join(snp_path, file)
                        datasets.append({
                            'order': int(order.replace('order', '')),
                            'snps': snp_size,
                            'dataset_id': dataset_id,
                            'path': full_path
                        })
    
    return sorted(datasets, key=lambda x: (x['order'], x['snps'], x['dataset_id']))


def load_data_from_path(path):
    """Load dataset from full path."""
    data = np.load(path, allow_pickle=True)
    
    X_train_list, y_train_list = [], []
    metadata = data['metadata'][0]
    for i in range(metadata['num_clients']):
        X_train_list.append(data[f'client_{i}_X'])
        y_train_list.append(data[f'client_{i}_y'])
    
    X_train = np.vstack(X_train_list).astype(np.float32)
    y_train = np.concatenate(y_train_list).astype(np.int32)
    X_val = data['validation_X'].astype(np.float32)
    y_val = data['validation_y'].astype(np.int32)
    X_test = data['test_X'].astype(np.float32)
    y_test = data['test_y'].astype(np.int32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model5_optimized(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the optimized Model5 classifier using ET-800-d12."""
    # Combine train and validation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    start_time = time.time()
    
    # Best configuration from testing
    et = ExtraTreesClassifier(
        n_estimators=800, 
        max_depth=12, 
        min_samples_split=2,
        random_state=42, 
        n_jobs=1  # Use 1 to avoid memory issues
    )
    
    et.fit(X_train_full, y_train_full)
    training_time = time.time() - start_time
    
    y_pred = et.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    del et
    gc.collect()
    
    return {
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train_full),
        'test_samples': len(X_test),
        'num_features': X_train.shape[1]
    }


def save_results(all_results, output_dir='results/model5_results'):
    """Save results to CSV, JSON, and text summary files."""
    if os.path.exists('results'):
        base_dir = 'results/model5_results'
    elif os.path.exists('../results'):
        base_dir = '../results/model5_results'
    else:
        base_dir = 'results/model5_results'
    
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to CSV
    csv_path = os.path.join(base_dir, f'model5_results_v2_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Results saved to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(base_dir, f'model5_results_v2_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save summary text
    summary_path = os.path.join(base_dir, f'model5_summary_v2_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL5 EVALUATION RESULTS SUMMARY (V2 - Optimized)\n")
        f.write("="*80 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: Pure Epistasis\n")
        f.write(f"Best Approach: ExtraTrees (n=800, depth=12, mss=2)\n\n")
        
        if all_results:
            avg_acc = np.mean([r['test_accuracy'] for r in all_results])
            best_acc = max([r['test_accuracy'] for r in all_results])
            worst_acc = min([r['test_accuracy'] for r in all_results])
            
            f.write(f"Total datasets tested: {len(all_results)}\n")
            f.write(f"Average accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")
            f.write(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
            f.write(f"Worst accuracy: {worst_acc:.4f} ({worst_acc*100:.2f}%)\n\n")
            
            for order in [2, 3]:
                order_results = [r for r in all_results if r['order'] == order]
                if order_results:
                    order_avg = np.mean([r['test_accuracy'] for r in order_results])
                    f.write(f"Order {order} average: {order_avg:.4f} ({order_avg*100:.2f}%)\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("Per-dataset results:\n")
            f.write(f"{'Dataset':<45} {'Accuracy':<15}\n")
            f.write("-"*70 + "\n")
            for r in all_results:
                name = f"order{r['order']}/snps{r['snp_size']}/dataset_{r['dataset_id']}"
                status = "GOOD" if r['test_accuracy'] >= 0.60 else ""
                f.write(f"{name:<45} {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%) {status}\n")
    
    print(f"Summary saved to: {summary_path}")
    return csv_path, json_path, summary_path


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION - MODEL5 V2 (OPTIMIZED)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Best Approach: ExtraTrees (n=800, depth=12, mss=2)")
    print("="*80)
    
    all_datasets = find_all_model5_datasets()
    print(f"\nFound {len(all_datasets)} datasets")
    
    all_results = []
    
    for i, ds in enumerate(all_datasets):
        print(f"\n[{i+1}/{len(all_datasets)}] order{ds['order']}/snps{ds['snps']}/dataset_{ds['dataset_id']}", end=" ")
        
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_path(ds['path'])
            
            results = train_model5_optimized(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            results['model_name'] = 'model5'
            results['model_type'] = 'Pure Epistasis'
            results['order'] = ds['order']
            results['snp_size'] = ds['snps']
            results['dataset_id'] = ds['dataset_id']
            all_results.append(results)
            
            status = "GOOD" if results['test_accuracy'] >= 0.60 else ""
            print(f"-> {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%) {status}")
            
        except Exception as e:
            print(f"-> Error: {e}")
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - MODEL5 V2")
    print("="*80)
    
    if all_results:
        avg_acc = np.mean([r['test_accuracy'] for r in all_results])
        best_acc = max([r['test_accuracy'] for r in all_results])
        worst_acc = min([r['test_accuracy'] for r in all_results])
        
        print(f"\nTotal datasets: {len(all_results)}")
        print(f"Average: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"Best: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Worst: {worst_acc:.4f} ({worst_acc*100:.2f}%)")
        
        for order in [2, 3]:
            order_results = [r for r in all_results if r['order'] == order]
            if order_results:
                order_avg = np.mean([r['test_accuracy'] for r in order_results])
                good_count = sum(1 for r in order_results if r['test_accuracy'] >= 0.60)
                print(f"\nOrder {order}: avg={order_avg:.4f}, {good_count}/{len(order_results)} >= 60%")
        
        # Save results
        print("\nSaving results...")
        save_results(all_results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == '__main__':
    results = main()
