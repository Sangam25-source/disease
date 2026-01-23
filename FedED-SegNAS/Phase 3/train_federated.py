#!/usr/bin/env python3
"""
Federated Learning Training Script
===================================

Trains Fuzzy CNN using federated learning on all disease models.

Usage:
------
# Train single model
python experiments/train_federated.py --model model1 --snps 50 --rounds 50

# Train all models
python experiments/train_federated.py --all --rounds 50

# Quick test
python experiments/train_federated.py --model model1 --snps 50 --rounds 5 --quick

Author: FedED-SegNAS Project
Date: January 2025
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import argparse
import json
import time
from datetime import datetime
import sys
import os

# Import from Phase 3 folder
phase3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, phase3_dir)
from federated_learning import SimpleFederatedTrainer

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def find_datasets(model_name, snp_size=None):
    """Find all available datasets for a model"""
    datasets = []
    model_dir = f'data/processed/{model_name}'
    
    if not os.path.exists(model_dir):
        return []
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                
                # Extract order
                order = 2 if 'order2' in full_path else 3
                
                # Extract num_snps
                num_snps = None
                for part in full_path.split(os.sep):
                    if part.startswith('snps'):
                        num_snps = int(part.replace('snps', ''))
                        break
                
                if num_snps is None:
                    continue
                
                # Filter by SNP size if specified
                if snp_size and num_snps != snp_size:
                    continue
                
                # Extract dataset ID
                dataset_id = int(file.replace('dataset_', '').replace('.npz', ''))
                
                datasets.append((order, num_snps, dataset_id, full_path))
    
    return sorted(datasets)


def train_single_dataset(model_name, filepath, num_rounds=50,
                         clients_per_round=10, local_epochs=5, 
                         use_nas=False, nas_frequency=5, verbose=True):
    """
    Train federated model on a single dataset
    
    Parameters:
    -----------
    use_nas : bool
        Enable Neural Architecture Search
    nas_frequency : int
        Run NAS every N rounds
    
    Returns:
    --------
    result : dict
        Comprehensive results including history and final metrics
    """
    if verbose:
        print(f"\n{'#'*80}")
        print(f"TRAINING: {model_name}")
        print(f"Dataset: {filepath}")
        print(f"{'#'*80}\n")
    
    try:
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Get dataset info
        X_test = data['test_X']
        num_snps = X_test.shape[1]
        
        # Count clients
        num_clients = sum(1 for k in data.keys() if k.startswith('client_') and k.endswith('_X'))
        
        if verbose:
            print(f">> Dataset Info:")
            print(f"   SNPs: {num_snps}")
            print(f"   Clients: {num_clients}")
            print(f"   Test samples: {len(X_test)}")
        
        # Initialize federated trainer
        trainer = SimpleFederatedTrainer(
            num_snps=num_snps,
            num_clients=num_clients,
            learning_rate=0.001,
            use_nas=use_nas
        )
        
        # Train
        start_time = time.time()
        history = trainer.train(
            federated_data=data,
            num_rounds=num_rounds,
            clients_per_round=clients_per_round,
            local_epochs=local_epochs,
            nas_frequency=nas_frequency,
            verbose=verbose
        )
        training_time = time.time() - start_time
        
        # Evaluate
        X_test = data['test_X']
        y_test = data['test_y']
        test_results = trainer.evaluate(X_test, y_test)
        
        # Compile results
        result = {
            'model_name': model_name,
            'filepath': filepath,
            'num_snps': num_snps,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'clients_per_round': clients_per_round,
            'local_epochs': local_epochs,
            'use_nas': use_nas,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'history': history,
            'test_loss': test_results['test_loss'],
            'test_accuracy': test_results['test_accuracy'],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'total_communication_mb': sum(history['communication_overhead_mb']),
            'avg_communication_per_round_mb': np.mean(history['communication_overhead_mb']),
            'convergence': 'Yes' if test_results['test_accuracy'] > 0.7 else 'Needs More Rounds',
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add NAS info if enabled
        if use_nas:
            result['nas_searches'] = history.get('nas_searches', [])
            result['num_nas_searches'] = len(history.get('nas_searches', []))
            result['nas_frequency'] = nas_frequency
            result['nas_frequency'] = nas_frequency
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Test Accuracy: {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)")
            print(f"Best Val Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"Training Time: {training_time/60:.2f} minutes")
            print(f"Total Communication: {result['total_communication_mb']:.2f} MB")
            print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f">> Error: {str(e)}")
        return {
            'model_name': model_name,
            'filepath': filepath,
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def save_results(result, output_dir='results/federated_training'):
    """Save training results"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = result['model_name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_path = f'{output_dir}/{model_name}_federated_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f">> Results saved: {json_path}")
    
    # Save learning curves if training succeeded
    if result['status'] == 'SUCCESS' and 'history' in result:
        plot_learning_curves(result, output_dir, timestamp)


def plot_learning_curves(result, output_dir, timestamp):
    """Plot and save learning curves"""
    history = result['history']
    model_name = result['model_name']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy
    axes[0, 0].plot(history['rounds'], history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(history['rounds'], history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'{model_name}: Accuracy over Rounds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(history['rounds'], history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(history['rounds'], history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'{model_name}: Loss over Rounds')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Communication Overhead
    axes[1, 0].plot(history['rounds'], history['communication_overhead_mb'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Communication (MB)')
    axes[1, 0].set_title(f'{model_name}: Communication Overhead')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Round Time
    axes[1, 1].plot(history['rounds'], history['round_time_seconds'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title(f'{model_name}: Round Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = f'{output_dir}/{model_name}_learning_curves_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f">> Learning curves saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning Training for FedED-SegNAS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='model1',
                       help='Disease model to train (model1-model8)')
    parser.add_argument('--snps', type=int, default=50,
                       help='SNP size (50, 100, 500, 1000, 2000, 5000)')
    parser.add_argument('--rounds', type=int, default=100,  # Changed from 50 - Task 2
                       help='Number of communication rounds')
    parser.add_argument('--clients-per-round', type=int, default=15,  # Increased from 10 - Task 2
                       help='Number of clients to sample per round')
    parser.add_argument('--local-epochs', type=int, default=10,  # Increased from 5 - Task 2
                       help='Number of local training epochs per client')
    parser.add_argument('--use-nas', action='store_true',
                       help='Enable Neural Architecture Search')
    parser.add_argument('--nas-frequency', type=int, default=5,
                       help='Run NAS every N rounds (only if --use-nas is enabled)')
    parser.add_argument('--all', action='store_true',
                       help='Train on all models')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (1 dataset per model)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING TRAINING" + (" WITH NAS" if args.use_nas else ""))
    print("="*80)
    print(f"Configuration:")
    print(f"  >> Model: {args.model if not args.all else 'ALL'}")
    print(f"  >> SNP Size: {args.snps}")
    print(f"  >> Rounds: {args.rounds}")
    print(f"  >> Clients per round: {args.clients_per_round}")
    print(f"  >> Local epochs: {args.local_epochs}")
    print(f"  >> Use NAS: {args.use_nas}")
    if args.use_nas:
        print(f"  >> NAS frequency: every {args.nas_frequency} rounds")
    print(f"  >> Quick mode: {args.quick}")
    print("="*80 + "\n")
    
    # Determine which models to train
    if args.all:
        models_to_train = [f'model{i}' for i in range(1, 9)]
    else:
        models_to_train = [args.model]
    
    all_results = []
    
    for model_name in models_to_train:
        print(f"\n{'#'*80}")
        print(f"PROCESSING {model_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Find datasets
        datasets = find_datasets(model_name, args.snps)
        
        if not datasets:
            print(f">> No datasets found for {model_name} with {args.snps} SNPs")
            continue
        
        print(f">> Found {len(datasets)} dataset(s)")
        
        # In quick mode, use only first dataset
        if args.quick:
            datasets = datasets[:1]
            print(f">> Quick mode: Using only first dataset")
        
        # Train on each dataset
        for order, num_snps, dataset_id, filepath in datasets:
            result = train_single_dataset(
                model_name,
                filepath,
                num_rounds=args.rounds,
                clients_per_round=args.clients_per_round,
                local_epochs=args.local_epochs,
                use_nas=args.use_nas,
                nas_frequency=args.nas_frequency,
                verbose=True
            )
            
            if result['status'] == 'SUCCESS':
                all_results.append(result)
                save_results(result)
    
    # Save summary
    if all_results:
        summary_path = f'results/federated_training/summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n>> Summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total models trained: {len(all_results)}")
    if all_results:
        avg_acc = np.mean([r['test_accuracy'] for r in all_results])
        print(f"Average test accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
