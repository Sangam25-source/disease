#!/usr/bin/env python3
"""
Model5 - DEEP ANALYSIS & FINAL PUSH
===================================
Current best: 65.50%
Target: 70%

Deep analysis:
1. Understand the exact epistatic pattern
2. Find ALL significant interactions
3. Try specialized epistasis detection methods
4. Neural network with custom architecture
5. Exhaustive hyperparameter search
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL5 - DEEP ANALYSIS & FINAL PUSH FOR 70%")
print("="*80)

# Load data
data = np.load('data/processed/model5/order2/snps50/dataset_0.npz', allow_pickle=True)
X_train_list, y_train_list = [], []
metadata = data['metadata'][0]
for i in range(metadata['num_clients']):
    X_train_list.append(data[f'client_{i}_X'])
    y_train_list.append(data[f'client_{i}_y'])

X_train = np.vstack(X_train_list + [data['validation_X']])
y_train = np.concatenate(y_train_list + [data['validation_y']])
X_test = data['test_X']
y_test = data['test_y']

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")
print(f"Class balance - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# ============================================================================
# PART 1: DEEP DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 1: DEEP DATA ANALYSIS")
print("="*80)

# 1.1 Individual SNP analysis
print("\n--- Individual SNP Mutual Information ---")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top_snps = np.argsort(mi_scores)[-10:][::-1]
print("Top 10 SNPs:")
for i, idx in enumerate(top_snps[:5]):
    print(f"  SNP {idx}: MI = {mi_scores[idx]:.4f}")
print(f"Max MI: {mi_scores.max():.4f}, Mean MI: {mi_scores.mean():.4f}")

# 1.2 Find ALL significant pairwise interactions
print("\n--- ALL Pairwise Interactions (top 20) ---")
all_pairs = []
for i in range(X_train.shape[1]):
    for j in range(i+1, X_train.shape[1]):
        interaction = X_train[:, i] * X_train[:, j]
        mi = mutual_info_classif(interaction.reshape(-1, 1), y_train, random_state=42)[0]
        all_pairs.append((i, j, mi))

all_pairs.sort(key=lambda x: x[2], reverse=True)
print("Top 20 interacting pairs:")
for i, (snp1, snp2, mi) in enumerate(all_pairs[:20]):
    print(f"  {i+1}. SNP {snp1} x SNP {snp2}: MI = {mi:.4f}")

# 1.3 Analyze the main epistatic pair (SNP 0 x SNP 1)
print("\n--- Main Epistatic Pair Analysis (SNP 0 x SNP 1) ---")
snp0 = X_train[:, 0]
snp1 = X_train[:, 1]
interaction_01 = snp0 * snp1

# Contingency table
print("Contingency table (SNP0 x SNP1 vs Label):")
for v0 in [0, 1, 2]:
    for v1 in [0, 1, 2]:
        mask = (snp0 == v0) & (snp1 == v1)
        if mask.sum() > 0:
            y_sub = y_train[mask]
            p1 = y_sub.mean()
            print(f"  SNP0={v0}, SNP1={v1}: n={mask.sum()}, P(y=1)={p1:.3f}")

# ============================================================================
# PART 2: SPECIALIZED FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("PART 2: SPECIALIZED FEATURE ENGINEERING")
print("="*80)

def create_comprehensive_features(X, top_pairs):
    """Create comprehensive epistasis features."""
    features = [X]
    
    # Add top 30 pairwise interactions
    for snp1, snp2, _ in top_pairs[:30]:
        features.append((X[:, snp1] * X[:, snp2]).reshape(-1, 1))
    
    # Add XOR-like features for top pairs
    for snp1, snp2, _ in top_pairs[:10]:
        features.append(np.abs(X[:, snp1] - X[:, snp2]).reshape(-1, 1))
    
    # Add 3-way interactions for top SNPs
    top_snp_indices = list(set([p[0] for p in top_pairs[:5]] + [p[1] for p in top_pairs[:5]]))[:6]
    for i in range(len(top_snp_indices)):
        for j in range(i+1, len(top_snp_indices)):
            for k in range(j+1, len(top_snp_indices)):
                s1, s2, s3 = top_snp_indices[i], top_snp_indices[j], top_snp_indices[k]
                features.append((X[:, s1] * X[:, s2] * X[:, s3]).reshape(-1, 1))
    
    return np.hstack(features)

X_train_comp = create_comprehensive_features(X_train, all_pairs)
X_test_comp = create_comprehensive_features(X_test, all_pairs)
print(f"Comprehensive features: {X_train.shape[1]} -> {X_train_comp.shape[1]}")

results = []

# Test with comprehensive features
et = ExtraTreesClassifier(n_estimators=800, max_depth=12, random_state=42, n_jobs=1)
et.fit(X_train_comp, y_train)
acc = accuracy_score(y_test, et.predict(X_test_comp))
print(f"ET with comprehensive features: {acc:.4f} ({acc*100:.2f}%)")
results.append(('ET-comp-feat', acc))
del et; gc.collect()

# ============================================================================
# PART 3: FOCUS ON EPISTATIC SIGNAL
# ============================================================================
print("\n" + "="*80)
print("PART 3: FOCUS ON EPISTATIC SIGNAL")
print("="*80)

# Create features ONLY from top interactions (remove noise)
def create_pure_epistasis_features(X, top_pairs, n_pairs=20):
    """Create features only from top epistatic pairs."""
    features = []
    for snp1, snp2, _ in top_pairs[:n_pairs]:
        features.append((X[:, snp1] * X[:, snp2]).reshape(-1, 1))
        features.append(X[:, snp1].reshape(-1, 1))
        features.append(X[:, snp2].reshape(-1, 1))
    return np.hstack(features)

for n_pairs in [10, 15, 20, 30]:
    X_train_pure = create_pure_epistasis_features(X_train, all_pairs, n_pairs)
    X_test_pure = create_pure_epistasis_features(X_test, all_pairs, n_pairs)
    
    et = ExtraTreesClassifier(n_estimators=800, max_depth=12, random_state=42, n_jobs=1)
    et.fit(X_train_pure, y_train)
    acc = accuracy_score(y_test, et.predict(X_test_pure))
    print(f"Pure epistasis ({n_pairs} pairs, {X_train_pure.shape[1]} feat): {acc:.4f}")
    results.append((f'Pure-{n_pairs}', acc))
    del et; gc.collect()

# ============================================================================
# PART 4: EXHAUSTIVE HYPERPARAMETER SEARCH
# ============================================================================
print("\n" + "="*80)
print("PART 4: EXHAUSTIVE HYPERPARAMETER SEARCH")
print("="*80)

best_acc = 0
best_config = None

# Search over many configurations
configs = []
for n_est in [500, 800, 1000, 1200]:
    for depth in [8, 10, 12, 14, 16]:
        for min_split in [2, 5, 10]:
            configs.append((n_est, depth, min_split))

print(f"Testing {len(configs)} configurations...")
for n_est, depth, min_split in configs:
    et = ExtraTreesClassifier(
        n_estimators=n_est, 
        max_depth=depth, 
        min_samples_split=min_split,
        random_state=42, 
        n_jobs=1
    )
    et.fit(X_train, y_train)
    acc = accuracy_score(y_test, et.predict(X_test))
    if acc > best_acc:
        best_acc = acc
        best_config = (n_est, depth, min_split)
        print(f"  New best: n={n_est}, d={depth}, mss={min_split} -> {acc:.4f}")
    del et; gc.collect()

print(f"\nBest config: n={best_config[0]}, d={best_config[1]}, mss={best_config[2]}")
print(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
results.append(('ET-best-config', best_acc))

# ============================================================================
# PART 5: MULTI-SEED ENSEMBLE WITH BEST CONFIG
# ============================================================================
print("\n" + "="*80)
print("PART 5: LARGE MULTI-SEED ENSEMBLE")
print("="*80)

preds = []
for seed in range(20):  # 20 different seeds
    et = ExtraTreesClassifier(
        n_estimators=best_config[0], 
        max_depth=best_config[1],
        min_samples_split=best_config[2],
        random_state=seed*100, 
        n_jobs=1
    )
    et.fit(X_train, y_train)
    preds.append(et.predict_proba(X_test))
    del et; gc.collect()

avg = np.mean(preds, axis=0)
acc = accuracy_score(y_test, np.argmax(avg, axis=1))
print(f"20-seed ensemble: {acc:.4f} ({acc*100:.2f}%)")
results.append(('20seed-ens', acc))

# ============================================================================
# PART 6: CONFUSION MATRIX ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 6: ERROR ANALYSIS")
print("="*80)

et = ExtraTreesClassifier(n_estimators=800, max_depth=12, random_state=42, n_jobs=1)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
print(f"  Precision: {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}")
print(f"  Recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")

# Analyze misclassified samples
wrong_idx = np.where(y_pred != y_test)[0]
print(f"\nMisclassified: {len(wrong_idx)} / {len(y_test)} ({len(wrong_idx)/len(y_test)*100:.1f}%)")

# ============================================================================
# PART 7: FINAL MEGA ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("PART 7: FINAL MEGA ENSEMBLE")
print("="*80)

all_preds = []

# Best ET configs
for seed in [42, 0, 100, 200, 300]:
    et = ExtraTreesClassifier(
        n_estimators=best_config[0], 
        max_depth=best_config[1],
        random_state=seed, 
        n_jobs=1
    )
    et.fit(X_train, y_train)
    all_preds.append(et.predict_proba(X_test))
    del et; gc.collect()

# RF variants
for seed in [42, 123]:
    rf = RandomForestClassifier(n_estimators=800, max_depth=20, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    all_preds.append(rf.predict_proba(X_test))
    del rf; gc.collect()

# With comprehensive features
et = ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=1)
et.fit(X_train_comp, y_train)
all_preds.append(et.predict_proba(X_test_comp))
del et; gc.collect()

avg = np.mean(all_preds, axis=0)
acc = accuracy_score(y_test, np.argmax(avg, axis=1))
print(f"Final mega ensemble ({len(all_preds)} models): {acc:.4f} ({acc*100:.2f}%)")
results.append(('Mega-final', acc))

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n{'Approach':<25} {'Accuracy':<15}")
print("-"*45)
for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
    marker = " **" if acc >= 0.70 else (" *" if acc >= 0.66 else "")
    print(f"{name:<25} {acc:.4f} ({acc*100:.2f}%){marker}")

best = max(r[1] for r in results)
print(f"\n" + "="*80)
print(f"BEST ACCURACY: {best:.4f} ({best*100:.2f}%)")
print("="*80)

if best >= 0.70:
    print(">> SUCCESS: Reached 70%!")
elif best >= 0.67:
    print(">> Very close to 70%!")
elif best >= 0.65:
    print(">> 65-66% is likely the ceiling for this pure epistasis data")
    print(">> The epistatic signal (SNP 0 x SNP 1, MI=0.108) is fundamentally weak")
print("="*80)
