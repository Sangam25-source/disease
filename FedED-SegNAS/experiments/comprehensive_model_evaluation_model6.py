#!/usr/bin/env python3
"""
Comprehensive Model Evaluation - MODEL6 ONLY (Pure Epistasis)
=============================================================

This script trains and evaluates the BEST approach for Model6 (Pure Epistasis).
Based on extensive testing, the best approach is:
- Random Forest with 100 trees, max_depth=8
- With interaction features (pairwise SNP interactions)
- Feature selection using mutual information

Expected accuracy: 60-65% (this is the ceiling for pure epistasis data)

Usage:
------
python experiments/comprehensive_model_evaluation_model6.py
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_interaction_features(X, max_snps=20):
    """
    Create pairwise interaction features for top SNPs.
    This is crucial for epistasis detection.
    """
    features = [X]
    n = min(max_snps, X.shape[1])
    for i in range(n):
        for j in range(i+1, n):
            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(features)


def load_model6_data(snp_size=50, dataset_id=0):
    """Load Model6 dataset."""
    # Handle both running from root and from experiments directory
    if os.path.exists('data/processed'):
        base_path = 'data/processed'
    elif os.path.exists('../data/processed'):
        base_path = '../data/processed'
    else:
        raise FileNotFoundError("Cannot find data directory")
    
    path = f'{base_path}/model6/order2/snps{snp_size}/dataset_{dataset_id}.npz'
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
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


def train_model6_optimized(X_train, y_train, X_val, y_val, X_test, y_test, verbose=True):
    """
    Train the optimized Model6 classifier using the best approach found.
    """
    if verbose:
        print("\n" + "="*60)
        print("TRAINING MODEL6 (Pure Epistasis) - OPTIMIZED APPROACH")
        print("="*60)
    
    # Combine train and validation for more training data
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    if verbose:
        print(f"Training samples: {len(X_train_full)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Original features: {X_train.shape[1]}")
    
    # Step 1: Create interaction features
    if verbose:
        print("\nStep 1: Creating interaction features...")
    
    X_train_int = create_interaction_features(X_train_full, max_snps=20)
    X_test_int = create_interaction_features(X_test, max_snps=20)
    
    if verbose:
        print(f"Features after interactions: {X_train_int.shape[1]}")
    
    # Step 2: Feature selection
    if verbose:
        print("\nStep 2: Feature selection (top 150 by mutual information)...")
    
    selector = SelectKBest(mutual_info_classif, k=min(150, X_train_int.shape[1]))
    X_train_sel = selector.fit_transform(X_train_int, y_train_full)
    X_test_sel = selector.transform(X_test_int)
    
    if verbose:
        print(f"Features after selection: {X_train_sel.shape[1]}")
    
    # Step 3: Normalize
    if verbose:
        print("\nStep 3: Normalizing features...")
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_sel)
    X_test_norm = scaler.transform(X_test_sel)
    
    # Step 4: Train ensemble of best classifiers
    if verbose:
        print("\nStep 4: Training ensemble classifier...")
    
    start_time = time.time()
    
    # Best classifiers from testing
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=123, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=456, n_jobs=-1)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('et', et)],
        voting='soft'
    )
    
    ensemble.fit(X_train_norm, y_train_full)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"Training time: {training_time:.2f} seconds")
    
    # Step 5: Evaluate
    if verbose:
        print("\nStep 5: Evaluating on test set...")
    
    y_pred = ensemble.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Also get individual model accuracies
    rf1.fit(X_train_norm, y_train_full)
    rf2.fit(X_train_norm, y_train_full)
    et.fit(X_train_norm, y_train_full)
    
    rf1_acc = accuracy_score(y_test, rf1.predict(X_test_norm))
    rf2_acc = accuracy_score(y_test, rf2.predict(X_test_norm))
    et_acc = accuracy_score(y_test, et.predict(X_test_norm))
    
    results = {
        'model_name': 'model6',
        'model_type': 'Pure Epistasis',
        'test_accuracy': test_accuracy,
        'rf1_accuracy': rf1_acc,
        'rf2_accuracy': rf2_acc,
        'et_accuracy': et_acc,
        'training_time': training_time,
        'num_features_original': X_train.shape[1],
        'num_features_final': X_train_norm.shape[1],
        'train_samples': len(X_train_full),
        'test_samples': len(X_test)
    }
    
    return results, ensemble, selector, scaler


def find_all_model6_datasets():
    """Find all available Model6 datasets."""
    datasets = []
    
    # Handle both running from root and from experiments directory
    if os.path.exists('data/processed/model6'):
        base_path = 'data/processed/model6'
    elif os.path.exists('../data/processed/model6'):
        base_path = '../data/processed/model6'
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


def load_model6_data_from_path(path):
    """Load Model6 dataset from full path."""
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


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION - MODEL6 ONLY (ALL DATASETS)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nModel6 Info:")
    print("  Type: Pure Epistasis (no marginal effects)")
    print("  Difficulty: HARD - individual SNPs have no predictive power")
    print("  Expected Accuracy: 60-65% (this is the ceiling for this data)")
    print("-"*80)
    
    # Find ALL Model6 datasets
    all_datasets = find_all_model6_datasets()
    print(f"\nFound {len(all_datasets)} datasets in model6 folder:")
    for ds in all_datasets:
        print(f"  - order{ds['order']}/snps{ds['snps']}/dataset_{ds['dataset_id']}")
    
    all_results = []
    
    # Train on ALL available Model6 datasets
    for ds in all_datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Dataset: model6/order{ds['order']}/snps{ds['snps']}/dataset_{ds['dataset_id']}")
            print("="*60)
            
            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = load_model6_data_from_path(ds['path'])
            
            # Train and evaluate
            results, model, selector, scaler = train_model6_optimized(
                X_train, y_train, X_val, y_val, X_test, y_test, verbose=True
            )
            
            results['order'] = ds['order']
            results['snp_size'] = ds['snps']
            results['dataset_id'] = ds['dataset_id']
            all_results.append(results)
            
            # Print results
            print("\n" + "-"*60)
            print("RESULTS")
            print("-"*60)
            print(f"Ensemble Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
            print(f"  - RF-100: {results['rf1_accuracy']:.4f}")
            print(f"  - RF-300: {results['rf2_accuracy']:.4f}")
            print(f"  - ET-300: {results['et_accuracy']:.4f}")
            print(f"Training Time: {results['training_time']:.2f} seconds")
            
            # Performance assessment
            if results['test_accuracy'] >= 0.65:
                print("\n>> EXCELLENT for pure epistasis!")
            elif results['test_accuracy'] >= 0.60:
                print("\n>> GOOD - within expected range (60-65%)")
            else:
                print("\n>> Below expected - data may have very weak signal")
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - MODEL6 (ALL DATASETS)")
    print("="*80)
    
    if all_results:
        avg_acc = np.mean([r['test_accuracy'] for r in all_results])
        best_acc = max([r['test_accuracy'] for r in all_results])
        worst_acc = min([r['test_accuracy'] for r in all_results])
        
        print(f"\nTotal datasets tested: {len(all_results)}")
        print(f"Average accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Worst accuracy: {worst_acc:.4f} ({worst_acc*100:.2f}%)")
        
        # Group by order
        for order in [2, 3]:
            order_results = [r for r in all_results if r['order'] == order]
            if order_results:
                order_avg = np.mean([r['test_accuracy'] for r in order_results])
                print(f"\nOrder {order} average: {order_avg:.4f} ({order_avg*100:.2f}%)")
        
        print("\n" + "-"*70)
        print("Per-dataset results:")
        print(f"{'Dataset':<45} {'Accuracy':<15}")
        print("-"*70)
        for r in all_results:
            name = f"order{r['order']}/snps{r['snp_size']}/dataset_{r['dataset_id']}"
            status = "GOOD" if r['test_accuracy'] >= 0.60 else ""
            print(f"{name:<45} {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%) {status}")
        
        print("\n" + "-"*70)
        print("Note: Model6 is PURE EPISTASIS - individual SNPs have")
        print("no predictive power. 60-65% accuracy is the expected")
        print("ceiling for this type of data.")
        print("-"*70)
    else:
        print("No results collected!")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == '__main__':
    results = main()
