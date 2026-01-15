#!/usr/bin/env python3
"""
Deep Analysis of Model2 Dataset
===============================

Model2 is classified as "Weak Marginal" - meaning individual SNPs
have some (weak) predictive power, unlike pure epistasis models.

From previous testing:
- Model2 average accuracy: ~58% (from comprehensive evaluation)
- This is better than pure epistasis but still challenging
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP ANALYSIS: MODEL2 (Weak Marginal)")
print("="*80)

# Check available datasets
print("\nChecking available Model2 datasets...")
base_path = 'data/processed/model2'

available_datasets = []
for order in ['order2', 'order3']:
    order_path = os.path.join(base_path, order)
    if os.path.exists(order_path):
        for snp_folder in os.listdir(order_path):
            if snp_folder.startswith('snps'):
                snp_path = os.path.join(order_path, snp_folder)
                for f in os.listdir(snp_path):
                    if f.endswith('.npz'):
                        available_datasets.append(f"{order}/{snp_folder}/{f}")

print(f"Found {len(available_datasets)} datasets:")
for ds in available_datasets[:10]:
    print(f"  - {ds}")
if len(available_datasets) > 10:
    print(f"  ... and {len(available_datasets) - 10} more")

# Load Model2 data (order2/snps50 first)
data_path = 'data/processed/model2/order2/snps50/dataset_0.npz'
print(f"\nLoading: {data_path}")
data = np.load(data_path, allow_pickle=True)

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

# Combine train + val
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Class balance - Train: {np.bincount(y_train_full)}, Test: {np.bincount(y_test)}")

# ============================================================================
# 1. Individual SNP Analysis
# ============================================================================
print("\n" + "="*80)
print("1. INDIVIDUAL SNP ANALYSIS (Mutual Information)")
print("="*80)

mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=42)
top_snps = np.argsort(mi_scores)[-15:][::-1]

print("\nTop 15 SNPs by mutual information:")
for i, idx in enumerate(top_snps):
    print(f"  {i+1}. SNP {idx}: MI = {mi_scores[idx]:.4f}")

print(f"\nMax MI: {mi_scores.max():.4f}")
print(f"Mean MI: {mi_scores.mean():.4f}")
print(f"SNPs with MI > 0.01: {(mi_scores > 0.01).sum()}")
print(f"SNPs with MI > 0.02: {(mi_scores > 0.02).sum()}")

if mi_scores.max() < 0.02:
    print("\n>> WEAK individual SNP signal (similar to pure epistasis)")
elif mi_scores.max() < 0.05:
    print("\n>> MODERATE individual SNP signal")
else:
    print("\n>> STRONG individual SNP signal")

# ============================================================================
# 2. Pairwise Interaction Analysis
# ============================================================================
print("\n" + "="*80)
print("2. PAIRWISE INTERACTION ANALYSIS")
print("="*80)

print("Analyzing pairwise interactions for top 25 SNPs...")
test_snps = list(range(min(25, X_train_full.shape[1])))

best_pairs = []
for i, j in combinations(test_snps, 2):
    interaction = X_train_full[:, i] * X_train_full[:, j]
    mi = mutual_info_classif(interaction.reshape(-1, 1), y_train_full, random_state=42)[0]
    best_pairs.append((i, j, mi))

best_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop 15 SNP pairs by interaction MI:")
for i, (snp1, snp2, mi) in enumerate(best_pairs[:15]):
    print(f"  {i+1}. SNP {snp1} x SNP {snp2}: MI = {mi:.4f}")

# Compare individual vs interaction
print("\n--- Individual vs Interaction Signal ---")
print(f"Best individual SNP MI: {mi_scores.max():.4f}")
print(f"Best pairwise interaction MI: {best_pairs[0][2]:.4f}")

if best_pairs[0][2] > mi_scores.max() * 2:
    print(">> Epistatic interactions are STRONGER than individual effects")
else:
    print(">> Individual effects are comparable to interactions")

# ============================================================================
# 3. Baseline ML Performance
# ============================================================================
print("\n" + "="*80)
print("3. BASELINE ML PERFORMANCE")
print("="*80)

models = [
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)),
    ('Extra Trees', ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)),
    ('Extra Trees d12', ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=1)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, C=0.1)),
]

print(f"\n{'Model':<25} {'Test Acc':<12}")
print("-"*40)

best_acc = 0
best_model = None
for name, model in models:
    model.fit(X_train_full, y_train_full)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<25} {acc:.4f} ({acc*100:.2f}%)")
    if acc > best_acc:
        best_acc = acc
        best_model = name

print(f"\nBest baseline: {best_acc:.4f} ({best_model})")

# ============================================================================
# 4. Test Different SNP Sizes
# ============================================================================
print("\n" + "="*80)
print("4. TESTING DIFFERENT SNP SIZES")
print("="*80)

for snp_size in [50, 100, 500, 1000]:
    try:
        path = f'data/processed/model2/order2/snps{snp_size}/dataset_0.npz'
        d = np.load(path, allow_pickle=True)
        
        X_tr_list, y_tr_list = [], []
        meta = d['metadata'][0]
        for i in range(meta['num_clients']):
            X_tr_list.append(d[f'client_{i}_X'])
            y_tr_list.append(d[f'client_{i}_y'])
        
        X_tr = np.vstack(X_tr_list + [d['validation_X']])
        y_tr = np.concatenate(y_tr_list + [d['validation_y']])
        X_te = d['test_X']
        y_te = d['test_y']
        
        et = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=1)
        et.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, et.predict(X_te))
        
        print(f"SNPs {snp_size}: ET accuracy = {acc:.4f} ({acc*100:.2f}%)")
    except Exception as e:
        print(f"SNPs {snp_size}: Error - {e}")

# ============================================================================
# 5. Feature Importance Analysis
# ============================================================================
print("\n" + "="*80)
print("5. FEATURE IMPORTANCE (from Extra Trees)")
print("="*80)

et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=1)
et.fit(X_train_full, y_train_full)
importances = et.feature_importances_

top_imp = np.argsort(importances)[-10:][::-1]
print("\nTop 10 features by importance:")
for i, idx in enumerate(top_imp):
    print(f"  {i+1}. SNP {idx}: importance = {importances[idx]:.4f}")

# ============================================================================
# 6. Summary and Recommendations
# ============================================================================
print("\n" + "="*80)
print("6. SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"\nModel2 Characteristics:")
print(f"  - Type: Weak Marginal")
print(f"  - Best individual SNP MI: {mi_scores.max():.4f}")
print(f"  - Best pairwise MI: {best_pairs[0][2]:.4f}")
print(f"  - Baseline accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

print(f"\nKey findings:")
if mi_scores.max() > 0.02:
    print("  - Individual SNPs have some predictive power")
if best_pairs[0][2] > 0.05:
    print("  - Strong pairwise interactions exist")
    
print(f"\nRecommendations:")
print("  1. Try ExtraTrees with depth tuning (d=10-15)")
print("  2. Add interaction features for top SNP pairs")
print("  3. Feature selection based on importance")
print("  4. Ensemble methods")

print("="*80)
