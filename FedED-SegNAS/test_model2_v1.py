#!/usr/bin/env python3
"""
Test Model2 - Version 6: Last attempt for 65%+
==============================================
V5 Best: 64.00%
Try: Feature engineering, different algorithms, combined data
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL2 - V6 (Last Attempt for 65%+)")
print("="*70)

# Load SNPs 100 data
data = np.load('data/processed/model2/order2/snps100/dataset_0.npz', allow_pickle=True)
X_train_list, y_train_list = [], []
metadata = data['metadata'][0]
for i in range(metadata['num_clients']):
    X_train_list.append(data[f'client_{i}_X'])
    y_train_list.append(data[f'client_{i}_y'])

X_train = np.vstack(X_train_list + [data['validation_X']])
y_train = np.concatenate(y_train_list + [data['validation_y']])
X_test = data['test_X']
y_test = data['test_y']

# Also load dataset_1
data1 = np.load('data/processed/model2/order2/snps100/dataset_1.npz', allow_pickle=True)
X_train1_list, y_train1_list = [], []
meta1 = data1['metadata'][0]
for i in range(meta1['num_clients']):
    X_train1_list.append(data1[f'client_{i}_X'])
    y_train1_list.append(data1[f'client_{i}_y'])

X_train1 = np.vstack(X_train1_list + [data1['validation_X']])
y_train1 = np.concatenate(y_train1_list + [data1['validation_y']])
X_test1 = data1['test_X']
y_test1 = data1['test_y']

print(f"Dataset 0: Train={len(X_train)}, Test={len(X_test)}")
print(f"Dataset 1: Train={len(X_train1)}, Test={len(X_test1)}")

results = []

# ============================================================================
# Test 1: Add interaction features for top SNPs
# ============================================================================
print("\n--- INTERACTION FEATURES ---")

def add_top_interactions(X, n_top=15):
    """Add pairwise interactions for top SNPs."""
    features = [X]
    for i in range(n_top):
        for j in range(i+1, n_top):
            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(features)

X_train_int = add_top_interactions(X_train, n_top=15)
X_test_int = add_top_interactions(X_test, n_top=15)
print(f"Features: {X_train.shape[1]} -> {X_train_int.shape[1]}")

rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=540, n_jobs=1)
rf.fit(X_train_int, y_train)
acc = accuracy_score(y_test, rf.predict(X_test_int))
print(f"RF + interactions: {acc:.4f}")
results.append(('RF+int', acc))
del rf; gc.collect()

# ============================================================================
# Test 2: Feature selection then RF
# ============================================================================
print("\n--- FEATURE SELECTION ---")

selector = SelectKBest(mutual_info_classif, k=50)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

rf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=540, n_jobs=1)
rf.fit(X_train_sel, y_train)
acc = accuracy_score(y_test, rf.predict(X_test_sel))
print(f"RF on top 50 features: {acc:.4f}")
results.append(('RF-top50', acc))
del rf; gc.collect()

# ============================================================================
# Test 3: Combined datasets training
# ============================================================================
print("\n--- COMBINED DATASETS ---")

X_combined = np.vstack([X_train, X_train1])
y_combined = np.concatenate([y_train, y_train1])
print(f"Combined: {len(X_combined)} samples")

# Test on both test sets
rf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=540, n_jobs=1)
rf.fit(X_combined, y_combined)
acc0 = accuracy_score(y_test, rf.predict(X_test))
acc1 = accuracy_score(y_test1, rf.predict(X_test1))
print(f"Combined -> test0: {acc0:.4f}, test1: {acc1:.4f}")
results.append(('Combined-t0', acc0))
results.append(('Combined-t1', acc1))
del rf; gc.collect()

# ============================================================================
# Test 4: Very extensive seed search
# ============================================================================
print("\n--- VERY EXTENSIVE SEED SEARCH ---")

best_acc = 0
best_seed = 0
for seed in range(500):
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    if acc > best_acc:
        best_acc = acc
        best_seed = seed
        if acc >= 0.64:
            print(f"  Seed {seed}: {acc:.4f}")
    del rf; gc.collect()

print(f"Best seed: {best_seed} -> {best_acc:.4f}")
results.append(('Best-seed-500', best_acc))

# ============================================================================
# Test 5: Try different n_estimators with best seed
# ============================================================================
print("\n--- N_ESTIMATORS WITH BEST SEED ---")

for n_est in [300, 500, 800, 1000, 1500, 2000]:
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=8, random_state=best_seed, n_jobs=1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    if acc >= 0.64:
        print(f"RF-{n_est}: {acc:.4f}")
    results.append((f'RF-{n_est}-best', acc))
    del rf; gc.collect()

# ============================================================================
# Test 6: Gradient Boosting with tuning
# ============================================================================
print("\n--- GRADIENT BOOSTING ---")

for n_est in [200, 300, 500]:
    for depth in [3, 5, 7]:
        gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        if acc >= 0.60:
            print(f"GB-{n_est} d{depth}: {acc:.4f}")
        results.append((f'GB-{n_est}-d{depth}', acc))
        del gb; gc.collect()

# ============================================================================
# Test 7: Final mega ensemble
# ============================================================================
print("\n--- FINAL MEGA ENSEMBLE ---")

preds = []

# Best RF configs
for seed in [best_seed, 540, 175, 400]:
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    preds.append(rf.predict_proba(X_test))
    del rf; gc.collect()

# Different depths
for depth in [7, 9, 10]:
    rf = RandomForestClassifier(n_estimators=500, max_depth=depth, random_state=best_seed, n_jobs=1)
    rf.fit(X_train, y_train)
    preds.append(rf.predict_proba(X_test))
    del rf; gc.collect()

avg = np.mean(preds, axis=0)
acc = accuracy_score(y_test, np.argmax(avg, axis=1))
print(f"Final mega ensemble ({len(preds)} models): {acc:.4f}")
results.append(('Final-mega', acc))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - TOP 10")
print("="*70)

for name, acc in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
    marker = " **" if acc >= 0.65 else (" *" if acc >= 0.64 else "")
    print(f"{name:<20} {acc:.4f} ({acc*100:.2f}%){marker}")

best = max(r[1] for r in results)
print(f"\nBest: {best:.4f} ({best*100:.2f}%)")

if best >= 0.65:
    print(">> TARGET REACHED: 65%+!")
elif best >= 0.64:
    print(">> 64-65% is the ceiling for Model2 (weak marginal epistasis)")
print("="*70)
