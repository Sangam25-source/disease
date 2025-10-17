#!/usr/bin/env python3
"""
Data Loading Utilities
Functions for loading and preparing genomic datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import os


class DataLoader:
    """
    Unified data loader for GAMETES and real genomic datasets
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
    
    def load_gametes_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load GAMETES output file
        
        GAMETES output format:
        - Tab-separated or space-separated
        - Columns: SNP1, SNP2, ..., SNPN, Class
        - SNP encoding: 0, 1, 2 (genotype counts)
        - Class: 0 (control), 1 (case)
        
        Args:
            filepath: Path to GAMETES output file
        
        Returns:
            X: Feature matrix (num_samples, num_snps)
            y: Labels (num_samples,)
        """
        try:
            # Try reading as tab-separated
            data = pd.read_csv(filepath, sep='\t', header=None)
        except:
            try:
                # Try space-separated
                data = pd.read_csv(filepath, sep='\s+', header=None)
            except Exception as e:
                raise ValueError(f"Cannot read file {filepath}: {e}")
        
        # Last column is the class label
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(np.int32)
        
        return X, y
    
    def load_plink_data(self, bed_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load PLINK binary format (.bed, .bim, .fam)
        
        Args:
            bed_prefix: Path prefix to PLINK files (without extension)
        
        Returns:
            X: Genotype matrix
            y: Phenotype labels (from .fam file)
        """
        try:
            from pandas_plink import read_plink
        except ImportError:
            raise ImportError("pandas-plink not installed. Install with: pip install pandas-plink")
        
        # Read PLINK files
        (bim, fam, bed) = read_plink(bed_prefix)
        
        # Extract genotypes
        X = bed.compute().T  # Transpose to (samples, snps)
        
        # Extract phenotypes from .fam file
        # .fam format: FID IID PID MID SEX PHENOTYPE
        y = fam['trait'].values
        
        # Convert to numpy arrays
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        return X, y
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Validate and analyze loaded data
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Dictionary with data statistics
        """
        stats = {
            'num_samples': X.shape[0],
            'num_snps': X.shape[1],
            'num_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'missing_rate': np.isnan(X).sum() / X.size,
            'data_type': X.dtype,
        }
        
        # Check for invalid values
        if np.any(X < 0) or np.any(X > 2):
            stats['warnings'] = 'Data contains values outside [0, 2] range'
        
        return stats
    
    def scan_gametes_datasets(self, base_dir='data/simulated') -> List[Dict]:
        """
        Scan directory for GAMETES datasets
        
        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return datasets
        
        # Scan for dataset files
        for model_dir in base_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            for order_dir in model_dir.iterdir():
                if not order_dir.is_dir():
                    continue
                
                # Extract epistasis order
                order = int(order_dir.name.replace('order', ''))
                
                for snps_dir in order_dir.iterdir():
                    if not snps_dir.is_dir():
                        continue
                    
                    # Extract SNP count
                    num_snps = int(snps_dir.name.replace('snps', ''))
                    
                    # Find dataset files
                    dataset_files = list(snps_dir.glob('dataset_*.txt'))
                    
                    for dataset_file in dataset_files:
                        dataset_id = int(dataset_file.stem.replace('dataset_', ''))
                        
                        datasets.append({
                            'model': model_name,
                            'order': order,
                            'num_snps': num_snps,
                            'dataset_id': dataset_id,
                            'filepath': str(dataset_file)
                        })
        
        return datasets


def test_data_loader():
    """Test data loading functionality"""
    print("Testing DataLoader...")
    
    loader = DataLoader()
    
    # Scan for datasets
    datasets = loader.scan_gametes_datasets()
    print(f"Found {len(datasets)} datasets")
    
    if len(datasets) > 0:
        # Test loading first dataset
        first_dataset = datasets[0]
        print(f"\nLoading test dataset: {first_dataset['filepath']}")
        
        X, y = loader.load_gametes_data(first_dataset['filepath'])
        
        # Validate
        stats = loader.validate_data(X, y)
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ DataLoader test passed!")
    else:
        print("⚠️  No datasets found to test")


if __name__ == '__main__':
    test_data_loader()
