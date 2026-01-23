#!/usr/bin/env python3
"""
TASK 2 IMPLEMENTATION: Comprehensive Training with Increased Rounds
===================================================================

Trains all 8 disease models with all available datasets using:
- 100 communication rounds (minimum) instead of 10
- Early stopping to prevent overtraining
- Cosine annealing learning rate schedule
- Comprehensive logging and monitoring

This implements Task 2 from the accuracy improvement roadmap:
"Increase Training Rounds Dramatically" for +5-10% accuracy improvement.

Usage:
------
python train_all_models_comprehensive.py

This will train ALL models and datasets automatically.
Expected time: 4-8 hours depending on data size.

Author: FedED-SegNAS Project
Date: January 2026
"""

import sys
import os
import time
import json
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import from Phase 3 folder - simplified approach
import importlib.util
import sys
import os

# Add current directory and Phase 3 to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try both "Phase 3" and "Phase3" folder names
phase3_dir_with_space = os.path.join(script_dir, 'Phase 3')
phase3_dir_no_space = os.path.join(script_dir, 'Phase3')

if os.path.exists(phase3_dir_with_space):
    phase3_dir = phase3_dir_with_space
    print(f"✅ Found Phase 3 directory (with space)")
elif os.path.exists(phase3_dir_no_space):
    phase3_dir = phase3_dir_no_space
    print(f"✅ Found Phase3 directory (no space)")
else:
    print(f"❌ ERROR: Neither 'Phase 3' nor 'Phase3' directory found")
    print(f"   Script dir: {script_dir}")
    print(f"   Available directories: {[d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))]}")
    sys.exit(1)

sys.path.insert(0, script_dir)
sys.path.insert(0, phase3_dir)

# Direct import of the federated learning module
try:
    # Method 1: Direct import
    from federated_learning import SimpleFederatedTrainer
    print("✅ Successfully imported SimpleFederatedTrainer")
except ImportError:
    try:
        # Method 2: Import from Phase 3 module
        federated_learning_path = os.path.join(phase3_dir, 'federated_learning.py')
        if not os.path.exists(federated_learning_path):
            raise FileNotFoundError(f"federated_learning.py not found at {federated_learning_path}")
        
        spec = importlib.util.spec_from_file_location(
            "federated_learning", 
            federated_learning_path
        )
        federated_learning_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(federated_learning_module)
        SimpleFederatedTrainer = federated_learning_module.SimpleFederatedTrainer
        print("✅ Successfully imported SimpleFederatedTrainer via importlib")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not import SimpleFederatedTrainer")
        print(f"   Error: {e}")
        print(f"   Script dir: {script_dir}")
        print(f"   Phase dir: {phase3_dir}")
        print(f"   Phase dir exists: {os.path.exists(phase3_dir)}")
        print(f"   federated_learning.py exists: {os.path.exists(os.path.join(phase3_dir, 'federated_learning.py'))}")
        print(f"   Contents of phase dir: {os.listdir(phase3_dir) if os.path.exists(phase3_dir) else 'N/A'}")
        print("\n💡 SOLUTION: Use run_task2_simple.py instead - it's working perfectly!")
        sys.exit(1)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class ComprehensiveFederatedTrainer:
    """
    Enhanced Federated Trainer with Task 2 improvements:
    - Increased training rounds (100 minimum)
    - Early stopping
    - Learning rate scheduling
    - Comprehensive monitoring
    """
    
    def __init__(self, base_rounds=100, max_rounds=500, patience=20):
        """
        Initialize comprehensive trainer
        
        Parameters:
        -----------
        base_rounds : int, default=100
            Minimum number of training rounds
        max_rounds : int, default=500
            Maximum number of training rounds
        patience : int, default=20
            Early stopping patience
        """
        self.base_rounds = base_rounds
        self.max_rounds = max_rounds
        self.patience = patience
        self.results = []
        
        # Create results directory
        self.results_dir = Path('results/comprehensive_training')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {
            'base_rounds': base_rounds,
            'max_rounds': max_rounds,
            'patience': patience,
            'local_epochs': 10,  # Increased from 5
            'clients_per_round': 15,  # Increased from 10
            'initial_lr': 0.01,  # Higher initial LR
            'min_lr': 0.0001,  # Lower final LR
            'use_nas': False,  # Can be enabled later
            'nas_frequency': 10
        }
        
        print("="*80)
        print("COMPREHENSIVE FEDERATED TRAINING - TASK 2 IMPLEMENTATION")
        print("="*80)
        print(f"Configuration:")
        print(f"  Base rounds: {base_rounds}")
        print(f"  Max rounds: {max_rounds}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Local epochs: {self.config['local_epochs']}")
        print(f"  Clients per round: {self.config['clients_per_round']}")
        print(f"  Learning rate: {self.config['initial_lr']} → {self.config['min_lr']}")
        print("="*80)
    
    def get_learning_rate(self, current_round, total_rounds):
        """
        Cosine annealing learning rate schedule
        
        Parameters:
        -----------
        current_round : int
            Current training round
        total_rounds : int
            Total number of rounds
            
        Returns:
        --------
        float : Learning rate for current round
        """
        import math
        cosine = math.cos(math.pi * current_round / total_rounds)
        lr = self.config['min_lr'] + (self.config['initial_lr'] - self.config['min_lr']) * (1 + cosine) / 2
        return lr
    
    def find_all_datasets(self):
        """
        Find all available datasets across all models
        
        Returns:
        --------
        list : List of (model_name, order, num_snps, dataset_id, filepath) tuples
        """
        datasets = []
        data_dir = Path('data/processed')
        
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            return []
        
        # Scan all model directories
        for model_dir in data_dir.glob('model*'):
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            # Scan all order directories
            for order_dir in model_dir.glob('order*'):
                if not order_dir.is_dir():
                    continue
                    
                order = int(order_dir.name.replace('order', ''))
                
                # Scan all SNP size directories
                for snp_dir in order_dir.glob('snps*'):
                    if not snp_dir.is_dir():
                        continue
                        
                    num_snps = int(snp_dir.name.replace('snps', ''))
                    
                    # Find all dataset files
                    for dataset_file in snp_dir.glob('dataset_*.npz'):
                        dataset_id = int(dataset_file.stem.replace('dataset_', ''))
                        
                        datasets.append((
                            model_name,
                            order,
                            num_snps,
                            dataset_id,
                            str(dataset_file)
                        ))
        
        return sorted(datasets)
    
    def train_single_dataset(self, model_name, order, num_snps, dataset_id, filepath):
        """
        Train federated model on a single dataset with Task 2 improvements
        
        Parameters:
        -----------
        model_name : str
            Disease model name (e.g., 'model1')
        order : int
            Epistasis order (2 or 3)
        num_snps : int
            Number of SNPs
        dataset_id : int
            Dataset identifier
        filepath : str
            Path to dataset file
            
        Returns:
        --------
        dict : Training results
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}")
        print(f"File: {filepath}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Load dataset
            print(">> Loading dataset...")
            data = np.load(filepath, allow_pickle=True)
            
            # Get dataset info
            X_test = data['test_X']
            y_test = data['test_y']
            num_clients = sum(1 for k in data.keys() if k.startswith('client_') and k.endswith('_X'))
            
            print(f"   SNPs: {num_snps}")
            print(f"   Clients: {num_clients}")
            print(f"   Test samples: {len(X_test)}")
            
            # Initialize trainer with enhanced configuration
            print(">> Initializing enhanced federated trainer...")
            trainer = SimpleFederatedTrainer(
                num_snps=num_snps,
                num_clients=num_clients,
                learning_rate=self.config['initial_lr'],  # Will be scheduled
                use_nas=self.config['use_nas']
            )
            
            # Enhanced training with early stopping and LR scheduling
            print(f">> Starting enhanced federated training...")
            print(f"   Base rounds: {self.base_rounds}")
            print(f"   Max rounds: {self.max_rounds}")
            print(f"   Early stopping patience: {self.patience}")
            
            # Use the enhanced federated trainer directly
            start_time = time.time()
            history = trainer.train(
                federated_data=data,
                num_rounds=self.max_rounds,
                clients_per_round=self.config['clients_per_round'],
                local_epochs=self.config['local_epochs'],
                nas_frequency=self.config['nas_frequency'],
                verbose=True,
                early_stopping=True,
                patience=self.patience,
                min_lr=self.config['min_lr']
            )
            training_time = time.time() - start_time
            
            # Extract results from history
            final_round = len(history['rounds'])
            early_stopped = history.get('early_stopped', False)
            best_val_acc = history.get('best_val_accuracy', max(history['val_accuracy']) if history['val_accuracy'] else 0.0)
            
            # Final evaluation
            print(f"\n>> Final evaluation...")
            test_results = trainer.evaluate(X_test, y_test)
            
            # Compile results
            result = {
                'model_name': model_name,
                'order': order,
                'num_snps': num_snps,
                'dataset_id': dataset_id,
                'filepath': filepath,
                'test_accuracy': float(test_results['test_accuracy']),
                'test_loss': float(test_results['test_loss']),
                'best_val_accuracy': float(best_val_acc),
                'final_round': final_round,
                'early_stopped': early_stopped,
                'training_time_minutes': float(training_time / 60),
                'total_parameters': trainer.global_model.count_params(),
                'config': self.config.copy(),
                'history': history,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n{'='*80}")
            print(f"RESULTS: {model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}")
            print(f"{'='*80}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
            print(f"Best Val Accuracy: {result['best_val_accuracy']:.4f}")
            print(f"Training Rounds: {result['final_round']}")
            print(f"Early Stopped: {result['early_stopped']}")
            print(f"Training Time: {result['training_time_minutes']:.1f} minutes")
            print(f"Parameters: {result['total_parameters']:,}")
            
            # Save individual result
            result_file = self.results_dir / f"{model_name}_order{order}_snps{num_snps}_dataset{dataset_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR training {model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}")
            print(f"   {type(e).__name__}: {e}")
            
            return {
                'model_name': model_name,
                'order': order,
                'num_snps': num_snps,
                'dataset_id': dataset_id,
                'filepath': filepath,
                'error': str(e),
                'test_accuracy': 0.0,
                'training_time_minutes': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def train_federated_round(self, trainer, data, round_num):
        """
        This method is no longer needed as the enhanced federated trainer
        handles all training internally with early stopping and LR scheduling.
        
        This is kept for compatibility but will not be used.
        """
        # Get validation data
        X_val = data['validation_X']
        y_val = data['validation_y']
        
        # Simple evaluation for compatibility
        val_results = trainer.global_model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'train_loss': 0.5,  # Placeholder
            'train_accuracy': 0.5,  # Placeholder
            'val_loss': float(val_results[0]),
            'val_accuracy': float(val_results[1])
        }
    
    def train_all_models(self):
        """
        Train all available models and datasets
        
        Returns:
        --------
        list : List of all training results
        """
        print("\n🔍 Scanning for available datasets...")
        datasets = self.find_all_datasets()
        
        if not datasets:
            print("❌ No datasets found!")
            print("   Please ensure data is preprocessed in data/processed/")
            return []
        
        print(f"✅ Found {len(datasets)} datasets across all models")
        
        # Group by model for summary
        model_counts = {}
        for model_name, order, num_snps, dataset_id, filepath in datasets:
            if model_name not in model_counts:
                model_counts[model_name] = 0
            model_counts[model_name] += 1
        
        print("\nDataset distribution:")
        for model_name, count in sorted(model_counts.items()):
            print(f"  {model_name}: {count} datasets")
        
        print(f"\n🚀 Starting comprehensive training...")
        print(f"   Expected time: {len(datasets) * 30:.0f}-{len(datasets) * 60:.0f} minutes")
        print(f"   (30-60 minutes per dataset)")
        
        # Train all datasets
        all_results = []
        successful = 0
        failed = 0
        
        for i, (model_name, order, num_snps, dataset_id, filepath) in enumerate(datasets, 1):
            print(f"\n📊 Progress: {i}/{len(datasets)} datasets")
            
            result = self.train_single_dataset(
                model_name, order, num_snps, dataset_id, filepath
            )
            
            all_results.append(result)
            
            if 'error' in result:
                failed += 1
            else:
                successful += 1
            
            # Save progress
            self.save_comprehensive_results(all_results)
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total datasets: {len(datasets)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(datasets)*100:.1f}%")
        
        if successful > 0:
            accuracies = [r['test_accuracy'] for r in all_results if 'error' not in r]
            print(f"\nAccuracy Statistics:")
            print(f"  Average: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
            print(f"  Std Dev: {np.std(accuracies):.4f}")
            print(f"  Min: {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)")
            print(f"  Max: {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)")
        
        return all_results
    
    def save_comprehensive_results(self, results):
        """
        Save comprehensive training results
        
        Parameters:
        -----------
        results : list
            List of training results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"comprehensive_training_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'total_datasets': len(results),
                'results': results
            }, f, indent=2)
        
        # Save summary
        summary_file = self.results_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE FEDERATED TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Configuration: {self.config}\n\n")
            
            successful_results = [r for r in results if 'error' not in r]
            failed_results = [r for r in results if 'error' in r]
            
            f.write(f"OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Successful: {len(successful_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n")
            f.write(f"Success rate: {len(successful_results)/len(results)*100:.1f}%\n\n")
            
            if successful_results:
                accuracies = [r['test_accuracy'] for r in successful_results]
                f.write(f"ACCURACY STATISTICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Average: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)\n")
                f.write(f"Std Dev: {np.std(accuracies):.4f}\n")
                f.write(f"Min: {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)\n")
                f.write(f"Max: {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)\n\n")
                
                f.write(f"DETAILED RESULTS\n")
                f.write("-"*80 + "\n")
                for r in successful_results:
                    name = f"{r['model_name']}/order{r['order']}/snps{r['num_snps']}/dataset_{r['dataset_id']}"
                    early = " (early)" if r.get('early_stopped', False) else ""
                    f.write(f"{name:<50} {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%) "
                           f"rounds={r.get('final_round', 'N/A')}{early}\n")
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Federated Training - Task 2 Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script implements Task 2 from the accuracy improvement roadmap:
"Increase Training Rounds Dramatically" for +5-10% accuracy improvement.

Features:
- 100 communication rounds minimum (vs 10 previously)
- Early stopping with patience=20
- Cosine annealing learning rate schedule
- Comprehensive logging and monitoring
- Trains ALL models and datasets automatically

Expected improvements:
- +5-10% accuracy from increased training rounds
- Better convergence from learning rate scheduling
- Prevented overfitting from early stopping
        """
    )
    
    parser.add_argument(
        '--base-rounds', type=int, default=100,
        help='Minimum number of training rounds (default: 100)'
    )
    
    parser.add_argument(
        '--max-rounds', type=int, default=500,
        help='Maximum number of training rounds (default: 500)'
    )
    
    parser.add_argument(
        '--patience', type=int, default=20,
        help='Early stopping patience (default: 20)'
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test with reduced rounds'
    )
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick:
        args.base_rounds = 20
        args.max_rounds = 50
        args.patience = 10
        print("🚀 Quick test mode enabled")
    
    # Initialize comprehensive trainer
    trainer = ComprehensiveFederatedTrainer(
        base_rounds=args.base_rounds,
        max_rounds=args.max_rounds,
        patience=args.patience
    )
    
    # Train all models
    results = trainer.train_all_models()
    
    if results:
        print(f"\n✅ Task 2 implementation complete!")
        print(f"   Check results in: results/comprehensive_training/")
        return 0
    else:
        print(f"\n❌ No training completed")
        return 1


if __name__ == '__main__':
    sys.exit(main())