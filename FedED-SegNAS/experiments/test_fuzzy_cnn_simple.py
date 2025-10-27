#!/usr/bin/env python3
"""
Simple Fuzzy CNN Test
=====================
Quick functional test to verify Fuzzy CNN implementation.
"""

import sys
sys.path.insert(0, '/app/FedED-SegNAS')

import tensorflow as tf
import numpy as np
from models.fuzzy_cnn import build_fuzzy_cnn

def test_fuzzy_cnn():
    print("="*60)
    print("TESTING FUZZY CNN")
    print("="*60)
    
    # Load test data
    data = np.load('data/processed/model1/order2/snps50/dataset_0.npz', allow_pickle=True)
    X_val = data['validation_X']
    y_val = data['validation_y']
    
    num_snps = X_val.shape[1]
    print(f"\nDataset: {len(X_val)} samples, {num_snps} SNPs")
    
    # Build model
    model = build_fuzzy_cnn(num_snps=num_snps)
    print("\nModel Summary:")
    model.summary()
    
    # Test training
    print("\nTraining for 10 epochs...")
    history = model.fit(
        X_val[:200].astype(np.float32), 
        y_val[:200],
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    final_acc = history.history['accuracy'][-1]
    print(f"\nFinal accuracy: {final_acc:.4f}")
    print("✅ Fuzzy CNN test PASSED!")

if __name__ == '__main__':
    test_fuzzy_cnn()
