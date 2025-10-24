#!/usr/bin/env python3
"""
Preprocess model1 and model5 datasets for Phase 1 completion
"""

import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, '/app/FedED-SegNAS')
from utils.data_loader import DataLoader


class DataPreprocessor:
    """
    Preprocesses GAMETES datasets for federated learning
    """
    
    def __init__(self, num_clients=50, test_split=0.15, val_split=0.15):
        self.num_clients = num_clients
        self.test_split = test_split
        self.val_split = val_split
        self.loader = DataLoader()
    
    def create_federated_splits(self, X, y):
        """
        Split data into federated clients + validation + test sets
        """
        # First split: separate test set
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_split,
            stratify=y,
            random_state=42
        )
        
        # Second split: separate validation set from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=self.val_split / (1 - self.test_split),
            stratify=y_trainval,
            random_state=42
        )
        
        # Distribute training data to clients (IID distribution)
        client_data = []
        samples_per_client = len(X_train) // self.num_clients
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else len(X_train)
            
            client_data.append({
                'client_id': i,
                'X': X_train[start_idx:end_idx],
                'y': y_train[start_idx:end_idx]
            })
        
        return {
            'clients': client_data,
            'validation': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test},
            'metadata': {
                'num_clients': self.num_clients,
                'num_train_samples': len(X_train),
                'num_val_samples': len(X_val),
                'num_test_samples': len(X_test),
                'num_features': X.shape[1],
                'class_distribution': {
                    'train': dict(zip(*np.unique(y_train, return_counts=True))),
                    'val': dict(zip(*np.unique(y_val, return_counts=True))),
                    'test': dict(zip(*np.unique(y_test, return_counts=True)))
                }
            }
        }
    
    def preprocess_single_dataset(self, dataset_info):
        """
        Preprocess a single dataset
        """
        try:
            # Load raw data
            X, y = self.loader.load_gametes_data(dataset_info['filepath'])
            
            # Create federated splits
            federated_data = self.create_federated_splits(X, y)
            
            # Create output directory
            output_dir = Path('data/processed') / dataset_info['model'] / f"order{dataset_info['order']}" / f"snps{dataset_info['num_snps']}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            output_file = output_dir / f"dataset_{dataset_info['dataset_id']}.npz"
            
            # Prepare data for saving
            save_dict = {
                'validation_X': federated_data['validation']['X'],
                'validation_y': federated_data['validation']['y'],
                'test_X': federated_data['test']['X'],
                'test_y': federated_data['test']['y'],
                'metadata': np.array([federated_data['metadata']], dtype=object)
            }
            
            # Add client data
            for i, client in enumerate(federated_data['clients']):
                save_dict[f'client_{i}_X'] = client['X']
                save_dict[f'client_{i}_y'] = client['y']
            
            np.savez_compressed(output_file, **save_dict)
            
            return True, str(output_file), federated_data['metadata']
            
        except Exception as e:
            return False, str(e), None
    
    def preprocess_models(self, model_names=['model1', 'model5']):
        """
        Preprocess specific models only
        """
        print("="*70)
        print("Data Preprocessing for model1 and model5")
        print("="*70)
        
        # Scan all datasets
        all_datasets = self.loader.scan_gametes_datasets()
        
        # Filter for specified models
        datasets = [d for d in all_datasets if d['model'] in model_names]
        
        print(f"\nConfiguration:")
        print(f"  Models to process: {', '.join(model_names)}")
        print(f"  Total datasets found: {len(datasets)}")
        print(f"  Federated clients: {self.num_clients}")
        print(f"  Test split: {self.test_split*100:.1f}%")
        print(f"  Validation split: {self.val_split*100:.1f}%")
        print("="*70)
        
        # Group by model
        model_stats = {}
        for model in model_names:
            model_datasets = [d for d in datasets if d['model'] == model]
            model_stats[model] = len(model_datasets)
            print(f"\n{model}: {len(model_datasets)} datasets")
        
        # Process each dataset
        successful = 0
        failed = 0
        total_train_samples = 0
        total_val_samples = 0
        total_test_samples = 0
        
        results_by_model = {model: {'success': 0, 'failed': 0} for model in model_names}
        
        print("\n" + "="*70)
        print("Processing datasets...")
        print("="*70)
        
        for dataset_info in tqdm(datasets, desc="Preprocessing"):
            success, result, metadata = self.preprocess_single_dataset(dataset_info)
            
            model = dataset_info['model']
            
            if success:
                successful += 1
                results_by_model[model]['success'] += 1
                if metadata:
                    total_train_samples += metadata['num_train_samples']
                    total_val_samples += metadata['num_val_samples']
                    total_test_samples += metadata['num_test_samples']
            else:
                failed += 1
                results_by_model[model]['failed'] += 1
                print(f"\n  ❌ Failed: {dataset_info['filepath']}")
                print(f"     Error: {result}")
        
        print("\n" + "="*70)
        print("Preprocessing Complete!")
        print("="*70)
        
        for model in model_names:
            print(f"\n{model}:")
            print(f"  ✅ Successfully processed: {results_by_model[model]['success']}/{model_stats[model]}")
            if results_by_model[model]['failed'] > 0:
                print(f"  ❌ Failed: {results_by_model[model]['failed']}")
        
        print(f"\nOverall:")
        print(f"  Total processed: {successful}/{len(datasets)} datasets")
        print(f"  Failed: {failed}")
        
        print(f"\nTotal samples across all datasets:")
        print(f"  Training: {total_train_samples:,}")
        print(f"  Validation: {total_val_samples:,}")
        print(f"  Test: {total_test_samples:,}")
        print(f"  Total: {total_train_samples + total_val_samples + total_test_samples:,}")
        print(f"\nProcessed data saved in: data/processed/")
        print("="*70)
        
        return successful, failed


def main():
    os.chdir('/app/FedED-SegNAS')
    
    # Initialize preprocessor with same config as model4
    preprocessor = DataPreprocessor(
        num_clients=50,
        test_split=0.15,
        val_split=0.15
    )
    
    # Preprocess model1 and model5
    successful, failed = preprocessor.preprocess_models(model_names=['model1', 'model5'])
    
    if successful > 0:
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📁 Preprocessed data saved in: data/processed/")
        return 0
    else:
        print(f"\n❌ Preprocessing failed!")
        return 1


if __name__ == '__main__':
    exit(main())
