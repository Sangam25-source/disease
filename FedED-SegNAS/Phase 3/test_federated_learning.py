"""
Unit tests for Federated Learning Framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from models.federated_learning import SimpleFederatedTrainer


def test_initialization():
    """Test SimpleFederatedTrainer initialization"""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)
    
    trainer = SimpleFederatedTrainer(num_snps=50, num_clients=50)
    
    assert trainer.num_snps == 50
    assert trainer.num_clients == 50
    assert trainer.global_model is not None
    
    print(">> Initialization test passed")


def test_weight_aggregation():
    """Test FedAvg weight aggregation"""
    print("\n" + "="*70)
    print("TEST 2: Weight Aggregation")
    print("="*70)
    
    trainer = SimpleFederatedTrainer(num_snps=50)
    
    # Create dummy weights
    weights1 = [np.ones((3, 3)) * 1.0, np.ones(3) * 1.0]
    weights2 = [np.ones((3, 3)) * 2.0, np.ones(3) * 2.0]
    weights3 = [np.ones((3, 3)) * 3.0, np.ones(3) * 3.0]
    
    client_weights = [weights1, weights2, weights3]
    
    # Simple average (no sample counts)
    aggregated = trainer.aggregate_weights(client_weights)
    
    # Should be average: (1+2+3)/3 = 2.0
    assert np.allclose(aggregated[0], 2.0)
    assert np.allclose(aggregated[1], 2.0)
    
    print(">> Weight aggregation test passed")


def test_client_training():
    """Test single client training"""
    print("\n" + "="*70)
    print("TEST 3: Client Training")
    print("="*70)
    
    trainer = SimpleFederatedTrainer(num_snps=50)
    
    # Create dummy client data
    client_data = {
        'X': np.random.randint(0, 3, (100, 50)).astype(np.float32),
        'y': np.random.randint(0, 2, 100).astype(np.int32)
    }
    
    global_weights = trainer.global_model.get_weights()
    
    # Train client
    local_weights, metrics = trainer.train_client(
        client_data, 
        global_weights, 
        local_epochs=2
    )
    
    assert len(local_weights) == len(global_weights)
    assert 'final_loss' in metrics
    assert 'final_accuracy' in metrics
    
    print(">> Client training test passed")


def test_full_federated_training():
    """Test complete federated training loop"""
    print("\n" + "="*70)
    print("TEST 4: Full Federated Training (Small Scale)")
    print("="*70)
    
    # Create synthetic federated data
    num_clients = 5
    num_snps = 50
    samples_per_client = 50
    
    federated_data = {}
    
    # Create client data
    for i in range(num_clients):
        federated_data[f'client_{i}_X'] = np.random.randint(0, 3, (samples_per_client, num_snps)).astype(np.float32)
        federated_data[f'client_{i}_y'] = np.random.randint(0, 2, samples_per_client).astype(np.int32)
    
    # Create validation data
    federated_data['validation_X'] = np.random.randint(0, 3, (100, num_snps)).astype(np.float32)
    federated_data['validation_y'] = np.random.randint(0, 2, 100).astype(np.int32)
    
    # Create test data
    federated_data['test_X'] = np.random.randint(0, 3, (100, num_snps)).astype(np.float32)
    federated_data['test_y'] = np.random.randint(0, 2, 100).astype(np.int32)
    
    # Initialize trainer
    trainer = SimpleFederatedTrainer(num_snps=num_snps, num_clients=num_clients)
    
    # Train for 3 rounds
    history = trainer.train(
        federated_data,
        num_rounds=3,
        clients_per_round=2,
        local_epochs=2,
        verbose=True
    )
    
    # Check history
    assert len(history['rounds']) == 3
    assert len(history['val_accuracy']) == 3
    assert 'communication_overhead_mb' in history
    
    # Evaluate
    results = trainer.evaluate(federated_data['test_X'], federated_data['test_y'])
    assert 'test_accuracy' in results
    
    print(f"\n  Final test accuracy: {results['test_accuracy']:.4f}")
    print(">> Full federated training test passed")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FEDERATED LEARNING FRAMEWORK - UNIT TESTS")
    print("="*80)
    
    test_initialization()
    test_weight_aggregation()
    test_client_training()
    test_full_federated_training()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80 + "\n")
