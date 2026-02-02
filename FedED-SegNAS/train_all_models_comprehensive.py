#!/usr/bin/env python3
"""
TASK 2 IMPLEMENTATION: Comprehensive Training with Increased Rounds - IMPROVED VERSION
=======================================================================================

Trains all 8 disease models with all available datasets using:
- Properly enforced minimum rounds
- Optimized hyperparameters for higher accuracy
- FedProx for client drift reduction
- Learning rate warmup + cosine annealing
- Robust early stopping with smoothing
- Comprehensive logging and monitoring

Key improvements over original:
- base_rounds is now properly enforced
- Reduced local_epochs to prevent client drift
- Better hyperparameter configuration
- Improved error handling and recovery
- Detailed per-model analysis

Usage:
------
python train.py                    # Train all models with default settings
python train.py --quick            # Quick test with reduced rounds
python train.py --model model1     # Train specific model only
python train.py --high-accuracy    # Use high-accuracy configuration

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
import traceback

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import from Phase 3 folder - simplified approach
import importlib.util

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
SimpleFederatedTrainer = None

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
        traceback.print_exc()
        sys.exit(1)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class TrainingConfiguration:
    """
    Centralized training configuration with presets for different scenarios.
    """
    
    # Default configuration - balanced accuracy and speed
    DEFAULT = {
        'min_rounds': 50,           # Minimum rounds (enforced!)
        'max_rounds': 200,          # Maximum rounds
        'patience': 25,             # Early stopping patience
        'local_epochs': 3,          # Reduced from 10 to prevent client drift
        'clients_per_round': 12,    # Balanced client sampling
        'initial_lr': 0.005,        # Moderate initial LR
        'min_lr': 0.00005,          # Lower final LR for fine-tuning
        'warmup_rounds': 10,        # Warmup before decay
        'use_warmup': True,         # Enable warmup
        'use_nas': False,           # NAS disabled by default
        'nas_frequency': 15,        # NAS frequency if enabled
        'use_fedprox': True,        # Enable FedProx for drift reduction
        'fedprox_mu': 0.01,         # FedProx proximal term coefficient
    }
    
    # High accuracy configuration - slower but better results
    HIGH_ACCURACY = {
        'min_rounds': 100,          # Higher minimum rounds
        'max_rounds': 500,          # Higher maximum rounds
        'patience': 40,             # More patience for convergence
        'local_epochs': 2,          # Lower to reduce drift further
        'clients_per_round': 15,    # More clients per round
        'initial_lr': 0.003,        # Lower initial LR for stability
        'min_lr': 0.00001,          # Very low final LR
        'warmup_rounds': 15,        # Longer warmup
        'use_warmup': True,
        'use_nas': False,
        'nas_frequency': 20,
        'use_fedprox': True,
        'fedprox_mu': 0.005,        # Lower mu for less aggressive regularization
    }
    
    # Quick test configuration - fast but lower accuracy
    QUICK_TEST = {
        'min_rounds': 10,
        'max_rounds': 30,
        'patience': 10,
        'local_epochs': 2,
        'clients_per_round': 8,
        'initial_lr': 0.01,
        'min_lr': 0.001,
        'warmup_rounds': 3,
        'use_warmup': True,
        'use_nas': False,
        'nas_frequency': 10,
        'use_fedprox': True,
        'fedprox_mu': 0.01,
    }
    
    # NAS-enabled configuration
    WITH_NAS = {
        'min_rounds': 80,
        'max_rounds': 300,
        'patience': 30,
        'local_epochs': 3,
        'clients_per_round': 12,
        'initial_lr': 0.005,
        'min_lr': 0.00005,
        'warmup_rounds': 15,
        'use_warmup': True,
        'use_nas': True,
        'nas_frequency': 20,
        'use_fedprox': True,
        'fedprox_mu': 0.01,
    }
    
    @classmethod
    def get_config(cls, preset='default'):
        """Get configuration by preset name."""
        presets = {
            'default': cls.DEFAULT,
            'high_accuracy': cls.HIGH_ACCURACY,
            'quick': cls.QUICK_TEST,
            'with_nas': cls.WITH_NAS,
        }
        return presets.get(preset, cls.DEFAULT).copy()


class ComprehensiveFederatedTrainer:
    """
    Enhanced Federated Trainer with all improvements:
    - Properly enforced minimum rounds
    - Optimized hyperparameters
    - FedProx for client drift reduction
    - Comprehensive monitoring and logging
    - Automatic model saving and recovery
    """
    
    def __init__(self, config=None, preset='default'):
        """
        Initialize comprehensive trainer.
        
        Parameters:
        -----------
        config : dict, optional
            Custom configuration dictionary
        preset : str, default='default'
            Configuration preset ('default', 'high_accuracy', 'quick', 'with_nas')
        """
        # Load configuration
        if config is None:
            self.config = TrainingConfiguration.get_config(preset)
        else:
            # Merge custom config with defaults
            self.config = TrainingConfiguration.get_config(preset)
            self.config.update(config)
        
        self.results = []
        self.failed_datasets = []
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f'results/comprehensive_training_{timestamp}')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models directory for saving trained models
        self.models_dir = self.results_dir / 'trained_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_configuration()
    
    def _print_configuration(self):
        """Print current configuration."""
        print("="*80)
        print("COMPREHENSIVE FEDERATED TRAINING - IMPROVED VERSION")
        print("="*80)
        print(f"Results directory: {self.results_dir}")
        print(f"\nConfiguration:")
        print(f"  Training Rounds:")
        print(f"    - Minimum rounds: {self.config['min_rounds']} (ENFORCED)")
        print(f"    - Maximum rounds: {self.config['max_rounds']}")
        print(f"    - Early stopping patience: {self.config['patience']}")
        print(f"  Client Training:")
        print(f"    - Local epochs: {self.config['local_epochs']}")
        print(f"    - Clients per round: {self.config['clients_per_round']}")
        print(f"  Learning Rate:")
        print(f"    - Initial LR: {self.config['initial_lr']}")
        print(f"    - Minimum LR: {self.config['min_lr']}")
        print(f"    - Warmup rounds: {self.config['warmup_rounds']} (enabled: {self.config['use_warmup']})")
        print(f"  FedProx:")
        print(f"    - Enabled: {self.config['use_fedprox']}")
        print(f"    - Mu: {self.config['fedprox_mu']}")
        print(f"  NAS:")
        print(f"    - Enabled: {self.config['use_nas']}")
        print(f"    - Frequency: {self.config['nas_frequency']}")
        print("="*80)
    
    def find_all_datasets(self, model_filter=None):
        """
        Find all available datasets across all models.
        
        Parameters:
        -----------
        model_filter : str, optional
            Filter to specific model (e.g., 'model1')
        
        Returns:
        --------
        list : List of (model_name, order, num_snps, dataset_id, filepath) tuples
        """
        datasets = []
        data_dir = Path('data/processed')
        
        if not data_dir.exists():
            print(f"❌ ERROR: Data directory not found: {data_dir}")
            return []
        
        # Scan all model directories
        for model_dir in sorted(data_dir.glob('model*')):
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            # Apply model filter if specified
            if model_filter is not None and model_name != model_filter:
                continue
            
            # Scan all order directories
            for order_dir in sorted(model_dir.glob('order*')):
                if not order_dir.is_dir():
                    continue
                    
                order = int(order_dir.name.replace('order', ''))
                
                # Scan all SNP size directories
                for snp_dir in sorted(order_dir.glob('snps*')):
                    if not snp_dir.is_dir():
                        continue
                        
                    num_snps = int(snp_dir.name.replace('snps', ''))
                    
                    # Find all dataset files
                    for dataset_file in sorted(snp_dir.glob('dataset_*.npz')):
                        dataset_id = int(dataset_file.stem.replace('dataset_', ''))
                        
                        datasets.append((
                            model_name,
                            order,
                            num_snps,
                            dataset_id,
                            str(dataset_file)
                        ))
        
        return datasets
    
    def validate_dataset(self, filepath):
        """
        Validate a dataset file before training.
        
        Parameters:
        -----------
        filepath : str
            Path to dataset file
            
        Returns:
        --------
        tuple : (is_valid, info_dict)
        """
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Check required keys
            required_keys = ['validation_X', 'validation_y', 'test_X', 'test_y']
            missing_keys = [k for k in required_keys if k not in data.keys()]
            
            if missing_keys:
                return False, {'error': f"Missing keys: {missing_keys}"}
            
            # Count clients
            num_clients = sum(
                1 for k in data.keys() 
                if k.startswith('client_') and k.endswith('_X')
            )
            
            if num_clients == 0:
                return False, {'error': "No client data found"}
            
            # Get data info
            info = {
                'num_clients': num_clients,
                'num_snps': data['test_X'].shape[1] if len(data['test_X'].shape) > 1 else 1,
                'test_samples': len(data['test_X']),
                'validation_samples': len(data['validation_X']),
                'classes': np.unique(data['test_y']).tolist()
            }
            
            # Check for class imbalance
            unique, counts = np.unique(data['test_y'], return_counts=True)
            imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            info['class_imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio > 5:
                print(f"    ⚠️ Warning: High class imbalance ratio: {imbalance_ratio:.2f}")
            
            return True, info
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def train_single_dataset(self, model_name, order, num_snps, dataset_id, filepath):
        """
        Train federated model on a single dataset with all improvements.
        
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
        dataset_name = f"{model_name}/order{order}/snps{num_snps}/dataset_{dataset_id}"
        
        print(f"\n{'='*80}")
        print(f"TRAINING: {dataset_name}")
        print(f"File: {filepath}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Validate dataset
            print(">> Validating dataset...")
            is_valid, info = self.validate_dataset(filepath)
            
            if not is_valid:
                raise ValueError(f"Invalid dataset: {info.get('error', 'Unknown error')}")
            
            print(f"   ✅ Dataset validated")
            print(f"   SNPs: {info['num_snps']}")
            print(f"   Clients: {info['num_clients']}")
            print(f"   Test samples: {info['test_samples']}")
            print(f"   Validation samples: {info['validation_samples']}")
            print(f"   Classes: {info['classes']}")
            print(f"   Class imbalance ratio: {info['class_imbalance_ratio']:.2f}")
            
            # Load dataset
            print(">> Loading dataset...")
            data = np.load(filepath, allow_pickle=True)
            
            X_test = data['test_X']
            y_test = data['test_y']
            
            # Initialize trainer with improved configuration
            print(">> Initializing improved federated trainer...")
            trainer = SimpleFederatedTrainer(
                num_snps=info['num_snps'],
                num_clients=info['num_clients'],
                learning_rate=self.config['initial_lr'],
                use_nas=self.config['use_nas'],
                use_fedprox=self.config['use_fedprox'],
                fedprox_mu=self.config['fedprox_mu']
            )
            
            # Start training with enforced minimum rounds
            print(f"\n>> Starting federated training...")
            print(f"   Minimum rounds: {self.config['min_rounds']} (ENFORCED)")
            print(f"   Maximum rounds: {self.config['max_rounds']}")
            print(f"   Early stopping patience: {self.config['patience']}")
            print(f"   FedProx enabled: {self.config['use_fedprox']}")
            
            training_start_time = time.time()
            
            # Call train with enforced minimum rounds
            history = trainer.train(
                federated_data=data,
                num_rounds=self.config['max_rounds'],
                clients_per_round=self.config['clients_per_round'],
                local_epochs=self.config['local_epochs'],
                nas_frequency=self.config['nas_frequency'],
                verbose=True,
                early_stopping=True,
                patience=self.config['patience'],
                min_lr=self.config['min_lr'],
                min_rounds=self.config['min_rounds'],  # ENFORCED minimum rounds
                warmup_rounds=self.config['warmup_rounds'],
                use_warmup=self.config['use_warmup']
            )
            
            training_time = time.time() - training_start_time
            
            # Final evaluation on test set
            print(f"\n>> Final evaluation on test set...")
            test_results = trainer.evaluate(X_test, y_test, restore_best=True)
            
            # Get training summary
            training_summary = trainer.get_training_summary()
            
            # Compile results
            result = {
                'model_name': model_name,
                'order': order,
                'num_snps': num_snps,
                'dataset_id': dataset_id,
                'filepath': filepath,
                'dataset_info': info,
                'test_accuracy': float(test_results['test_accuracy']),
                'test_loss': float(test_results['test_loss']),
                'per_class_accuracy': test_results.get('per_class_accuracy', {}),
                'precision': test_results.get('precision', None),
                'recall': test_results.get('recall', None),
                'f1_score': test_results.get('f1_score', None),
                'best_val_accuracy': float(history['best_val_accuracy']),
                'final_val_accuracy': float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0.0,
                'final_round': len(history['rounds']),
                'early_stopped': history.get('early_stopped', False),
                'early_stop_round': history.get('early_stop_round', None),
                'training_time_minutes': float(training_time / 60),
                'total_parameters': trainer.global_model.count_params(),
                'avg_client_drift': float(np.mean(history.get('client_drift_metrics', [0]))),
                'config': self.config.copy(),
                'training_summary': training_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print results summary
            print(f"\n{'='*80}")
            print(f"RESULTS: {dataset_name}")
            print(f"{'='*80}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
            print(f"Best Val Accuracy: {result['best_val_accuracy']:.4f} ({result['best_val_accuracy']*100:.2f}%)")
            print(f"Training Rounds: {result['final_round']}")
            print(f"Early Stopped: {result['early_stopped']}", end="")
            if result['early_stopped']:
                print(f" (at round {result['early_stop_round']})")
            else:
                print()
            print(f"Training Time: {result['training_time_minutes']:.1f} minutes")
            print(f"Parameters: {result['total_parameters']:,}")
            print(f"Avg Client Drift: {result['avg_client_drift']:.4f}")
            
            if result['precision'] is not None:
                print(f"Precision: {result['precision']:.4f}")
                print(f"Recall: {result['recall']:.4f}")
                print(f"F1 Score: {result['f1_score']:.4f}")
            
            # Save individual result
            result_filename = f"{model_name}_order{order}_snps{num_snps}_dataset{dataset_id}_result.json"
            result_file = self.results_dir / result_filename
            
            # Save result without history (history can be large)
            result_to_save = {k: v for k, v in result.items() if k != 'history'}
            with open(result_file, 'w') as f:
                json.dump(result_to_save, f, indent=2, default=str)
            
            # Save trained model
            model_filename = f"{model_name}_order{order}_snps{num_snps}_dataset{dataset_id}_model.h5"
            model_file = self.models_dir / model_filename
            trainer.save_model(str(model_file))
            
            # Save training history
            history_filename = f"{model_name}_order{order}_snps{num_snps}_dataset{dataset_id}_history.json"
            history_file = self.results_dir / history_filename
            trainer.save_history(str(history_file))
            
            # Generate training plots
            self._generate_training_plots(history, dataset_name, model_name, order, num_snps, dataset_id)
            
            print(f"\n💾 Saved:")
            print(f"   Result: {result_file}")
            print(f"   Model: {model_file}")
            print(f"   History: {history_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"\n❌ ERROR training {dataset_name}")
            print(f"   {error_msg}")
            traceback.print_exc()
            
            self.failed_datasets.append({
                'dataset_name': dataset_name,
                'filepath': filepath,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'model_name': model_name,
                'order': order,
                'num_snps': num_snps,
                'dataset_id': dataset_id,
                'filepath': filepath,
                'error': error_msg,
                'test_accuracy': 0.0,
                'training_time_minutes': (time.time() - start_time) / 60,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_training_plots(self, history, dataset_name, model_name, order, num_snps, dataset_id):
        """Generate and save training visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Training Progress: {dataset_name}', fontsize=14)
            
            rounds = history['rounds']
            
            # Plot 1: Accuracy curves
            ax1 = axes[0, 0]
            ax1.plot(rounds, history['train_accuracy'], label='Train Accuracy', color='blue', alpha=0.7)
            ax1.plot(rounds, history['val_accuracy'], label='Val Accuracy', color='green', alpha=0.7)
            if history.get('smoothed_val_accuracy'):
                ax1.plot(rounds, history['smoothed_val_accuracy'], label='Smoothed Val Acc', 
                        color='red', linestyle='--', alpha=0.7)
            ax1.axhline(y=history['best_val_accuracy'], color='orange', linestyle=':', 
                       label=f'Best Val: {history["best_val_accuracy"]:.4f}')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Training & Validation Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss curves
            ax2 = axes[0, 1]
            ax2.plot(rounds, history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
            ax2.plot(rounds, history['val_loss'], label='Val Loss', color='green', alpha=0.7)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training & Validation Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Learning rate schedule
            ax3 = axes[1, 0]
            ax3.plot(rounds, history['learning_rate'], label='Learning Rate', color='purple')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Plot 4: Client drift
            ax4 = axes[1, 1]
            if history.get('client_drift_metrics'):
                ax4.plot(rounds, history['client_drift_metrics'], label='Client Drift', color='red')
                ax4.set_xlabel('Round')
                ax4.set_ylabel('Average Client Drift')
                ax4.set_title('Client Drift Over Training')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No drift data available', ha='center', va='center')
                ax4.set_title('Client Drift (N/A)')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{model_name}_order{order}_snps{num_snps}_dataset{dataset_id}_training.png"
            plot_file = self.plots_dir / plot_filename
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   Plot: {plot_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate plots: {e}")
    
    def train_all_models(self, model_filter=None):
        """
        Train all available models and datasets.
        
        Parameters:
        -----------
        model_filter : str, optional
            Filter to specific model (e.g., 'model1')
        
        Returns:
        --------
        list : List of all training results
        """
        print("\n🔍 Scanning for available datasets...")
        datasets = self.find_all_datasets(model_filter=model_filter)
        
        if not datasets:
            print("❌ No datasets found!")
            print("   Please ensure data is preprocessed in data/processed/")
            return []
        
        print(f"✅ Found {len(datasets)} datasets")
        
        if model_filter:
            print(f"   (filtered to: {model_filter})")
        
        # Group by model for summary
        model_counts = {}
        for model_name, order, num_snps, dataset_id, filepath in datasets:
            if model_name not in model_counts:
                model_counts[model_name] = []
            model_counts[model_name].append((order, num_snps, dataset_id))
        
        print("\nDataset distribution:")
        for model_name, items in sorted(model_counts.items()):
            print(f"  {model_name}: {len(items)} datasets")
            # Show breakdown by order
            order_counts = {}
            for order, num_snps, dataset_id in items:
                if order not in order_counts:
                    order_counts[order] = 0
                order_counts[order] += 1
            for order, count in sorted(order_counts.items()):
                print(f"    Order {order}: {count} datasets")
        
        # Estimate training time
        estimated_time_per_dataset = (
            self.config['min_rounds'] * 0.5 +  # Rough estimate: 0.5 min per round
            self.config['max_rounds'] * 0.1    # Additional time if running full
        )
        total_estimated_time = len(datasets) * estimated_time_per_dataset
        
        print(f"\n🚀 Starting comprehensive training...")
        print(f"   Estimated time: {total_estimated_time/60:.1f} - {total_estimated_time*1.5/60:.1f} hours")
        print(f"   (depends on early stopping and data size)")
        
        # Save training configuration
        config_file = self.results_dir / 'training_config.json'
        with open(config_file, 'w') as f:
            json.dump({
                'config': self.config,
                'num_datasets': len(datasets),
                'model_filter': model_filter,
                'start_time': datetime.now().isoformat()
            }, f, indent=2)
        
        # Train all datasets
        all_results = []
        successful = 0
        failed = 0
        
        training_start_time = time.time()
        
        for i, (model_name, order, num_snps, dataset_id, filepath) in enumerate(datasets, 1):
            print(f"\n{'#'*80}")
            print(f"📊 Progress: {i}/{len(datasets)} datasets ({i/len(datasets)*100:.1f}%)")
            print(f"{'#'*80}")
            
            result = self.train_single_dataset(
                model_name, order, num_snps, dataset_id, filepath
            )
            
            all_results.append(result)
            
            if 'error' in result:
                failed += 1
            else:
                successful += 1
            
            # Save progress after each dataset
            self._save_progress(all_results, successful, failed, training_start_time)
            
            # Print running statistics
            if successful > 0:
                current_accuracies = [r['test_accuracy'] for r in all_results if 'error' not in r]
                print(f"\n📈 Running Statistics:")
                print(f"   Completed: {successful} successful, {failed} failed")
                print(f"   Current avg accuracy: {np.mean(current_accuracies):.4f} ({np.mean(current_accuracies)*100:.2f}%)")
                print(f"   Best accuracy so far: {np.max(current_accuracies):.4f} ({np.max(current_accuracies)*100:.2f}%)")
        
        total_time = time.time() - training_start_time
        
        # Generate final summary
        self._generate_final_summary(all_results, successful, failed, total_time)
        
        return all_results
    
    def _save_progress(self, results, successful, failed, start_time):
        """Save intermediate progress."""
        progress_file = self.results_dir / 'training_progress.json'
        
        elapsed_time = time.time() - start_time
        
        progress = {
            'total_completed': len(results),
            'successful': successful,
            'failed': failed,
            'elapsed_time_minutes': elapsed_time / 60,
            'last_update': datetime.now().isoformat(),
            'results': results
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
    
    def _generate_final_summary(self, results, successful, failed, total_time):
        """Generate comprehensive final summary."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total datasets: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            accuracies = [r['test_accuracy'] for r in successful_results]
            
            print(f"\n📊 ACCURACY STATISTICS:")
            print(f"   Average: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
            print(f"   Std Dev: {np.std(accuracies):.4f}")
            print(f"   Min: {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)")
            print(f"   Max: {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)")
            print(f"   Median: {np.median(accuracies):.4f} ({np.median(accuracies)*100:.2f}%)")
            
            # Accuracy distribution
            print(f"\n📈 ACCURACY DISTRIBUTION:")
            bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            for low, high in bins:
                count = sum(1 for a in accuracies if low <= a < high)
                percentage = count / len(accuracies) * 100
                bar = '█' * int(percentage / 2)
                print(f"   {low:.1f}-{high:.1f}: {count:3d} ({percentage:5.1f}%) {bar}")
            
            # Per-model breakdown
            print(f"\n📋 PER-MODEL BREAKDOWN:")
            model_results = {}
            for r in successful_results:
                model = r['model_name']
                if model not in model_results:
                    model_results[model] = []
                model_results[model].append(r['test_accuracy'])
            
            for model, accs in sorted(model_results.items()):
                print(f"   {model}: avg={np.mean(accs):.4f}, "
                      f"min={np.min(accs):.4f}, max={np.max(accs):.4f}, n={len(accs)}")
            
            # Top 5 best results
            print(f"\n🏆 TOP 5 BEST RESULTS:")
            sorted_results = sorted(successful_results, key=lambda x: x['test_accuracy'], reverse=True)
            for i, r in enumerate(sorted_results[:5], 1):
                name = f"{r['model_name']}/order{r['order']}/snps{r['num_snps']}/dataset_{r['dataset_id']}"
                print(f"   {i}. {name}: {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%)")
            
            # Bottom 5 results (for analysis)
            print(f"\n⚠️ BOTTOM 5 RESULTS (for analysis):")
            for i, r in enumerate(sorted_results[-5:], 1):
                name = f"{r['model_name']}/order{r['order']}/snps{r['num_snps']}/dataset_{r['dataset_id']}"
                print(f"   {i}. {name}: {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%)")
        
        # Failed datasets
        if self.failed_datasets:
            print(f"\n❌ FAILED DATASETS ({len(self.failed_datasets)}):")
            for fd in self.failed_datasets:
                print(f"   - {fd['dataset_name']}: {fd['error']}")
        
        # Save comprehensive summary
        summary_file = self.results_dir / 'final_summary.json'
        summary = {
            'total_datasets': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0,
            'total_time_minutes': total_time / 60,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        if successful_results:
            summary['accuracy_stats'] = {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'median': float(np.median(accuracies))
            }
            summary['per_model_stats'] = {
                model: {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)),
                    'min': float(np.min(accs)),
                    'max': float(np.max(accs)),
                    'count': len(accs)
                }
                for model, accs in model_results.items()
            }
        
        summary['failed_datasets'] = self.failed_datasets
        summary['all_results'] = results
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save text summary
        text_summary_file = self.results_dir / 'final_summary.txt'
        with open(text_summary_file, 'w') as f:
            f.write("COMPREHENSIVE FEDERATED TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success rate: {successful/len(results)*100:.1f}%\n\n")
            
            if successful_results:
                f.write("ACCURACY STATISTICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Average: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)\n")
                f.write(f"Std Dev: {np.std(accuracies):.4f}\n")
                f.write(f"Min: {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)\n")
                f.write(f"Max: {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)\n")
                f.write(f"Median: {np.median(accuracies):.4f}\n\n")
                
                f.write("DETAILED RESULTS\n")
                f.write("-"*80 + "\n")
                for r in sorted(successful_results, key=lambda x: x['test_accuracy'], reverse=True):
                    name = f"{r['model_name']}/order{r['order']}/snps{r['num_snps']}/dataset_{r['dataset_id']}"
                    early = " (early)" if r.get('early_stopped', False) else ""
                    f.write(f"{name:<50} {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%) "
                           f"rounds={r.get('final_round', 'N/A')}{early}\n")
        
        print(f"\n💾 Results saved to: {self.results_dir}")
        print(f"   - final_summary.json")
        print(f"   - final_summary.txt")
        print(f"   - training_progress.json")
        print(f"   - Individual results, models, and plots")
        
        # Generate overall training plot
        self._generate_overall_summary_plot(successful_results)
    
    def _generate_overall_summary_plot(self, results):
        """Generate overall summary visualization."""
        if not results:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Comprehensive Training Summary', fontsize=14)
            
            accuracies = [r['test_accuracy'] for r in results]
            
            # Plot 1: Accuracy distribution histogram
            ax1 = axes[0, 0]
            ax1.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
            ax1.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(accuracies):.4f}')
            ax1.axvline(x=np.median(accuracies), color='green', linestyle='--', 
                       label=f'Median: {np.median(accuracies):.4f}')
            ax1.set_xlabel('Test Accuracy')
            ax1.set_ylabel('Count')
            ax1.set_title('Accuracy Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Per-model box plot
            ax2 = axes[0, 1]
            model_data = {}
            for r in results:
                model = r['model_name']
                if model not in model_data:
                    model_data[model] = []
                model_data[model].append(r['test_accuracy'])
            
            models = sorted(model_data.keys())
            data = [model_data[m] for m in models]
            ax2.boxplot(data, labels=models)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Test Accuracy')
            ax2.set_title('Accuracy by Model')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Training time distribution
            ax3 = axes[1, 0]
            times = [r['training_time_minutes'] for r in results]
            ax3.hist(times, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax3.axvline(x=np.mean(times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(times):.1f} min')
            ax3.set_xlabel('Training Time (minutes)')
            ax3.set_ylabel('Count')
            ax3.set_title('Training Time Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Accuracy vs Training Rounds
            ax4 = axes[1, 1]
            rounds = [r['final_round'] for r in results]
            colors = ['green' if not r.get('early_stopped', False) else 'blue' for r in results]
            ax4.scatter(rounds, accuracies, c=colors, alpha=0.6)
            ax4.set_xlabel('Training Rounds')
            ax4.set_ylabel('Test Accuracy')
            ax4.set_title('Accuracy vs Training Rounds (blue=early stopped)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.results_dir / 'overall_summary.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   - overall_summary.png")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate overall summary plot: {e}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Federated Training - Improved Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Train all models with default configuration
  python train.py
  
  # Train with high-accuracy configuration
  python train.py --high-accuracy
  
  # Quick test
  python train.py --quick
  
  # Train specific model only
  python train.py --model model1
  
  # Custom configuration
  python train.py --min-rounds 100 --max-rounds 300 --patience 30
  
  # Enable NAS
  python train.py --with-nas
        """
    )
    
    # Preset configurations
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        '--quick', action='store_true',
        help='Quick test with reduced rounds'
    )
    preset_group.add_argument(
        '--high-accuracy', action='store_true',
        help='High accuracy configuration (slower but better results)'
    )
    preset_group.add_argument(
        '--with-nas', action='store_true',
        help='Enable Neural Architecture Search'
    )
    
    # Custom configuration options
    parser.add_argument(
        '--min-rounds', type=int, default=None,
        help='Minimum number of training rounds (enforced)'
    )
    parser.add_argument(
        '--max-rounds', type=int, default=None,
        help='Maximum number of training rounds'
    )
    parser.add_argument(
        '--patience', type=int, default=None,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--local-epochs', type=int, default=None,
        help='Number of local epochs per client'
    )
    parser.add_argument(
        '--clients-per-round', type=int, default=None,
        help='Number of clients to train per round'
    )
    parser.add_argument(
        '--initial-lr', type=float, default=None,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--min-lr', type=float, default=None,
        help='Minimum learning rate'
    )
    parser.add_argument(
        '--no-fedprox', action='store_true',
        help='Disable FedProx (use standard FedAvg)'
    )
    parser.add_argument(
        '--fedprox-mu', type=float, default=None,
        help='FedProx proximal term coefficient'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Train specific model only (e.g., model1)'
    )
    
    args = parser.parse_args()
    
    # Determine preset
    if args.quick:
        preset = 'quick'
        print("🚀 Using QUICK TEST configuration")
    elif args.high_accuracy:
        preset = 'high_accuracy'
        print("🎯 Using HIGH ACCURACY configuration")
    elif args.with_nas:
        preset = 'with_nas'
        print("🔬 Using NAS-ENABLED configuration")
    else:
        preset = 'default'
        print("⚙️ Using DEFAULT configuration")
    
    # Build custom config from arguments
    custom_config = {}
    
    if args.min_rounds is not None:
        custom_config['min_rounds'] = args.min_rounds
    if args.max_rounds is not None:
        custom_config['max_rounds'] = args.max_rounds
    if args.patience is not None:
        custom_config['patience'] = args.patience
    if args.local_epochs is not None:
        custom_config['local_epochs'] = args.local_epochs
    if args.clients_per_round is not None:
        custom_config['clients_per_round'] = args.clients_per_round
    if args.initial_lr is not None:
        custom_config['initial_lr'] = args.initial_lr
    if args.min_lr is not None:
        custom_config['min_lr'] = args.min_lr
    if args.no_fedprox:
        custom_config['use_fedprox'] = False
    if args.fedprox_mu is not None:
        custom_config['fedprox_mu'] = args.fedprox_mu
    
    # Initialize trainer
    trainer = ComprehensiveFederatedTrainer(
        config=custom_config if custom_config else None,
        preset=preset
    )
    
    # Train all models (or filtered model)
    results = trainer.train_all_models(model_filter=args.model)
    
    if results:
        successful = sum(1 for r in results if 'error' not in r)
        if successful > 0:
            accuracies = [r['test_accuracy'] for r in results if 'error' not in r]
            print(f"\n✅ Training complete!")
            print(f"   Average accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
            print(f"   Results saved to: {trainer.results_dir}")
            return 0
        else:
            print(f"\n❌ All training attempts failed")
            return 1
    else:
        print(f"\n❌ No training completed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
