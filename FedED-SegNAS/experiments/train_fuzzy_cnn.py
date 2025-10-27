#!/usr/bin/env python3
"""
Single-Model Training Script for Fuzzy CNN
==========================================

Baseline training (non-federated) on one dataset.
Trains a Fuzzy CNN on combined data from all clients to establish performance baseline.

Usage:
------
python experiments/train_fuzzy_cnn.py
python experiments/train_fuzzy_cnn.py --model model5 --snps 100 --epochs 100

Author: FedED-SegNAS Project
Date: October 2024
"""

import sys
sys.path.insert(0, '/app/FedED-SegNAS')

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from models.fuzzy_cnn import build_fuzzy_cnn
import time
import json


def load_preprocessed_data(model_name, order, num_snps, dataset_id):
    """
    Load preprocessed dataset and combine all client data for centralized training.
    
    Parameters:
    -----------
    model_name : str
        Name of disease model (model1, model5, etc.)
    order : int
        Epistasis order (2 or 3)
    num_snps : int
        Number of SNP features
    dataset_id : int
        Dataset index
    
    Returns:
    --------
    tuple
        (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata
    """
    filepath = f'data/processed/{model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}.npz'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    
    # Combine all client data for centralized baseline training
    X_train = []
    y_train = []
    
    metadata = data['metadata'][0]
    num_clients = metadata['num_clients']
    
    print(f"Combining data from {num_clients} clients...")
    for i in range(num_clients):
        X_train.append(data[f'client_{i}_X'])
        y_train.append(data[f'client_{i}_y'])
    
    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.concatenate(y_train).astype(np.int32)
    
    X_val = data['validation_X'].astype(np.float32)
    y_val = data['validation_y'].astype(np.int32)
    X_test = data['test_X'].astype(np.float32)
    y_test = data['test_y'].astype(np.int32)
    
    print(f"✅ Data loaded:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   SNPs:  {X_train.shape[1]}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


def plot_training_history(history, save_path='results/plots/training_history.png'):
    """
    Plot training and validation metrics.
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history object
    save_path : str
        Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history plot saved to: {save_path}")


def train_single_model(model_name='model1', order=2, num_snps=50, 
                       dataset_id=0, epochs=50, batch_size=64, 
                       learning_rate=0.001, save_model=True):
    """
    Train Fuzzy CNN on single dataset (baseline, non-federated).
    
    Parameters:
    -----------
    model_name : str
        Disease model name
    order : int
        Epistasis order
    num_snps : int
        Number of SNPs
    dataset_id : int
        Dataset ID
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    save_model : bool
        Whether to save trained model
    
    Returns:
    --------
    dict
        Training results including history and test metrics
    """
    print("\n" + "=" * 70)
    print(f"TRAINING FUZZY CNN: {model_name} (order{order}, SNPs={num_snps})")
    print("=" * 70 + "\n")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_preprocessed_data(
        model_name, order, num_snps, dataset_id
    )
    
    # Build model
    print(f"\nBuilding Fuzzy CNN...")
    model = build_fuzzy_cnn(num_snps=num_snps, learning_rate=learning_rate)
    
    total_params = model.count_params()
    print(f"Model built with {total_params:,} parameters")
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    
    print(f"\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print("=" * 70 + "\n")
    
    # Save results
    results_dir = f'results/models/{model_name}/order{order}/snps{num_snps}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training history
    plot_path = f'{results_dir}/training_history_dataset{dataset_id}.png'
    plot_training_history(history, plot_path)
    
    # Save model
    if save_model:
        model_path = f'{results_dir}/fuzzy_cnn_dataset{dataset_id}.h5'
        model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
    
    # Save results JSON
    results_data = {
        'model_name': model_name,
        'order': order,
        'num_snps': num_snps,
        'dataset_id': dataset_id,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_parameters': int(total_params),
        'training_time_seconds': float(training_time),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    results_path = f'{results_dir}/results_dataset{dataset_id}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")
    
    return results_data


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train Fuzzy CNN on single dataset (baseline)'
    )
    
    parser.add_argument('--model', type=str, default='model1',
                       help='Disease model name (default: model1)')
    parser.add_argument('--order', type=int, default=2,
                       help='Epistasis order (default: 2)')
    parser.add_argument('--snps', type=int, default=50,
                       help='Number of SNPs (default: 50)')
    parser.add_argument('--dataset-id', type=int, default=0,
                       help='Dataset ID (default: 0)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained model')
    
    args = parser.parse_args()
    
    # Train
    results = train_single_model(
        model_name=args.model,
        order=args.order,
        num_snps=args.snps,
        dataset_id=args.dataset_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_model=not args.no_save
    )
    
    print("\n✅ Training complete!")
    print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")


if __name__ == '__main__':
    main()
