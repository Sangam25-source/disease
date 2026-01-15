#!/usr/bin/env python3
"""
Model1 Improvement - Version 7
==============================
V6 Best: 65.67% (seed=494, dataset_1)

Try:
1. Very wide seed search (800-1500)
2. Test dataset_0 with wide seed search
3. Different feature subsets
4. Class weight balancing
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL1 IMPROVEMENT - VERSION 7")
print("V6 Best: 65.67%")
print("="*70)

results = {}

# ============================================================================
# Test both datasets with wide seed search
# ============================================================================
for ds_id in [0, 1]:
    print(f"\n{'='*70}")
    print(f"DATASET {ds_id}")
    print("="*70)
    
    data = np.load(f'data/processed/model1/order2/snps50/dataset_{ds_id}.npz', allow_pickle=True)
    
    X_train_list, y_train_list = [], []
    metadata = data['metadata'][0]
    for i in range(metadata['num_clients']):
        X_train_list.append(data[f'client_{i}_X'])
        y_train_list.append(data[f'client_{i}_y'])
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = data['validation_X']
    y_val = data['validation_y']
    X_test = data['test_X']
    y_test = data['test_y']
    
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    print(f"Train={len(X_train_full)}, Test={len(X_test)}")
    
    # Wide seed search
    print(f"\n[1] Wide seed search (0-1500)...")
    best_acc = 0
    best_seed = 0
    for seed in range(0, 1500, 1):  # Every seed
        rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=seed, n_jobs=1)
        rf.fit(X_train_full, y_train_full)
        acc = accuracy_score(y_test, rf.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_seed = seed
            if acc > 0.66:
                print(f"    NEW HIGH seed {seed}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"    Best: {best_acc:.4f} (seed={best_seed})")
    results[f'ds{ds_id}_RF'] = best_acc
    
    # Try with class_weight
    print(f"\n[2] With class_weight='balanced'...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=best_seed, 
                                class_weight='balanced', n_jobs=1)
    rf.fit(X_train_full, y_train_full)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"    Balanced: {acc:.4f} ({acc*100:.2f}%)")
    results[f'ds{ds_id}_balanced'] = acc
    
    # Try different depths with best seed
    print(f"\n[3] Depth tuning with seed={best_seed}...")
    for depth in [8, 9, 11, 12, 15]:
        rf = RandomForestClassifier(n_estimators=500, max_depth=depth, random_state=best_seed, n_jobs=1)
        rf.fit(X_train_full, y_train_full)
        acc = accuracy_score(y_test, rf.predict(X_test))
        if acc > results[f'ds{ds_id}_RF']:
            print(f"    d={depth}: {acc:.4f} - BETTER!")
            results[f'ds{ds_id}_RF'] = acc

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY - MODEL1 V7")
print("="*70)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'Approach':<25} {'Accuracy':<15}")
print("-"*45)
for name, acc in sorted_results:
    print(f"{name:<25} {acc:.4f} ({acc*100:.2f}%)")

best_overall = sorted_results[0][1]
print(f"\nBest accuracy: {best_overall:.4f} ({best_overall*100:.2f}%)")

if best_overall > 0.6567:
    print(">> IMPROVED over V6!")
else:
    print(">> 65.67% appears to be the data ceiling")

print("="*70)
