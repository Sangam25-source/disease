#!/usr/bin/env python3
"""
Data Validation Script
Validates quality of preprocessed datasets for federated learning
"""

import numpy as np
import sys
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/app/FedED-SegNAS')


class DataValidator:
    def __init__(self, processed_dir='/app/FedED-SegNAS/data/processed'):
        self.processed_dir = processed_dir
        self.validation_results = []
        self.failed_checks = []
        
    def check_class_balance(self, y, threshold=0.4):
        """
        Check case-control balance
        Returns True if balance ratio >= threshold (default allows 60-40 split)
        """
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            return False, f"Expected 2 classes, found {len(unique)}"
        
        balance_ratio = min(counts) / max(counts)
        is_balanced = balance_ratio >= threshold
        
        return is_balanced, f"Balance ratio: {balance_ratio:.3f} (classes: {dict(zip(unique, counts))})"
    
    def check_missing_values(self, data_dict):
        """Check for missing values in all arrays"""
        missing_info = {}
        has_missing = False
        
        for key, arr in data_dict.items():
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                missing_count = np.isnan(arr).sum()
                if missing_count > 0:
                    has_missing = True
                    missing_info[key] = missing_count
        
        if has_missing:
            return False, f"Missing values found: {missing_info}"
        return True, "No missing values"
    
    def check_genotype_encoding(self, X):
        """Verify genotypes are in {0, 1, 2}"""
        unique_values = np.unique(X)
        valid_values = {0, 1, 2}
        
        if not all(val in valid_values for val in unique_values):
            invalid = [v for v in unique_values if v not in valid_values]
            return False, f"Invalid genotype values found: {invalid}"
        
        return True, f"Valid encoding: {sorted(unique_values.tolist())}"
    
    def check_data_shape(self, data_dict, metadata):
        """Validate data shape consistency"""
        issues = []
        
        # Check validation set
        expected_val = metadata['num_val_samples']
        actual_val = len(data_dict['validation_y'])
        if expected_val != actual_val:
            issues.append(f"Validation size mismatch: expected {expected_val}, got {actual_val}")
        
        # Check test set
        expected_test = metadata['num_test_samples']
        actual_test = len(data_dict['test_y'])
        if expected_test != actual_test:
            issues.append(f"Test size mismatch: expected {expected_test}, got {actual_test}")
        
        # Check feature count
        expected_features = metadata['num_features']
        actual_features = data_dict['validation_X'].shape[1]
        if expected_features != actual_features:
            issues.append(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
        
        if issues:
            return False, "; ".join(issues)
        return True, f"Shapes valid (val:{actual_val}, test:{actual_test}, features:{actual_features})"
    
    def check_federated_splits(self, data_dict, metadata):
        """Verify federated split integrity"""
        num_clients = metadata['num_clients']
        issues = []
        
        total_client_samples = 0
        for i in range(num_clients):
            key = f'client_{i}_X'
            if key not in data_dict:
                issues.append(f"Missing client {i}")
            else:
                total_client_samples += len(data_dict[key])
        
        expected_train = metadata['num_train_samples']
        if total_client_samples != expected_train:
            issues.append(f"Client sample mismatch: expected {expected_train}, got {total_client_samples}")
        
        if issues:
            return False, "; ".join(issues)
        return True, f"All {num_clients} clients present with {total_client_samples} total samples"
    
    def calculate_maf(self, X):
        """Calculate Minor Allele Frequency for each SNP"""
        # X shape: (samples, snps)
        # Genotypes: 0, 1, 2
        # MAF = (count_1 + 2*count_2) / (2*total_samples)
        
        maf_values = []
        for snp_idx in range(X.shape[1]):
            snp_col = X[:, snp_idx]
            allele_count = np.sum(snp_col)  # 0*count_0 + 1*count_1 + 2*count_2
            total_alleles = 2 * len(snp_col)
            maf = allele_count / total_alleles
            
            # MAF is the frequency of the minor (less common) allele
            maf = min(maf, 1 - maf)
            maf_values.append(maf)
        
        return np.array(maf_values)
    
    def validate_dataset(self, filepath):
        """Validate single preprocessed dataset"""
        dataset_name = str(Path(filepath).relative_to(self.processed_dir))
        result = {
            'dataset': dataset_name,
            'filepath': filepath,
            'checks': {},
            'passed': True
        }
        
        try:
            # Load data
            data = np.load(filepath, allow_pickle=True)
            metadata = data['metadata'][0]
            
            # Prepare data dict
            data_dict = {key: data[key] for key in data.keys()}
            
            # Check 1: Class balance (validation set)
            passed, msg = self.check_class_balance(data['validation_y'])
            result['checks']['class_balance'] = {'passed': passed, 'message': msg}
            if not passed:
                result['passed'] = False
                self.failed_checks.append(f"{dataset_name}: Class balance - {msg}")
            
            # Check 2: Missing values
            passed, msg = self.check_missing_values(data_dict)
            result['checks']['missing_values'] = {'passed': passed, 'message': msg}
            if not passed:
                result['passed'] = False
                self.failed_checks.append(f"{dataset_name}: Missing values - {msg}")
            
            # Check 3: Genotype encoding
            passed, msg = self.check_genotype_encoding(data['validation_X'])
            result['checks']['genotype_encoding'] = {'passed': passed, 'message': msg}
            if not passed:
                result['passed'] = False
                self.failed_checks.append(f"{dataset_name}: Genotype encoding - {msg}")
            
            # Check 4: Data shapes
            passed, msg = self.check_data_shape(data_dict, metadata)
            result['checks']['data_shape'] = {'passed': passed, 'message': msg}
            if not passed:
                result['passed'] = False
                self.failed_checks.append(f"{dataset_name}: Data shape - {msg}")
            
            # Check 5: Federated splits
            passed, msg = self.check_federated_splits(data_dict, metadata)
            result['checks']['federated_splits'] = {'passed': passed, 'message': msg}
            if not passed:
                result['passed'] = False
                self.failed_checks.append(f"{dataset_name}: Federated splits - {msg}")
            
            # Calculate MAF for statistics
            maf_values = self.calculate_maf(data['validation_X'])
            result['maf_mean'] = float(np.mean(maf_values))
            result['maf_std'] = float(np.std(maf_values))
            result['maf_values'] = maf_values
            
            # Store metadata
            result['metadata'] = metadata
            
        except Exception as e:
            result['passed'] = False
            result['error'] = str(e)
            self.failed_checks.append(f"{dataset_name}: Exception - {str(e)}")
        
        self.validation_results.append(result)
        return result
    
    def generate_visualizations(self):
        """Generate validation visualizations"""
        print("\n" + "="*70)
        print("Generating Visualizations")
        print("="*70)
        
        # Create output directory
        plot_dir = Path('results/plots')
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect statistics
        models = defaultdict(list)
        maf_by_model = defaultdict(list)
        
        for result in self.validation_results:
            if result['passed'] and 'metadata' in result:
                model_name = result['dataset'].split('/')[0]
                models[model_name].append(result)
                if 'maf_values' in result:
                    maf_by_model[model_name].extend(result['maf_values'])
        
        # Plot 1: MAF Distribution
        if maf_by_model:
            fig, axes = plt.subplots(1, len(maf_by_model), figsize=(6*len(maf_by_model), 5))
            if len(maf_by_model) == 1:
                axes = [axes]
            
            for idx, (model, maf_values) in enumerate(sorted(maf_by_model.items())):
                axes[idx].hist(maf_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                axes[idx].set_xlabel('Minor Allele Frequency', fontsize=11)
                axes[idx].set_ylabel('Count', fontsize=11)
                axes[idx].set_title(f'{model} MAF Distribution\n(mean={np.mean(maf_values):.3f})', 
                                   fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].axvline(np.mean(maf_values), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(maf_values):.3f}')
                axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'maf_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ MAF distribution plot saved")
        
        # Plot 2: Dataset Summary Statistics
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = []
        dataset_counts = []
        colors_list = []
        
        for model, results in sorted(models.items()):
            model_names.append(model)
            dataset_counts.append(len(results))
            colors_list.append('green' if len(results) > 0 else 'red')
        
        bars = ax.bar(range(len(model_names)), dataset_counts, color=colors_list, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Valid Datasets', fontsize=12, fontweight='bold')
        ax.set_title('Validated Datasets by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, dataset_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'dataset_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Dataset summary plot saved")
        
        # Plot 3: Class Balance Verification
        fig, ax = plt.subplots(figsize=(10, 6))
        
        balance_ratios = []
        dataset_labels = []
        
        for result in self.validation_results[:20]:  # Show first 20
            if result['passed'] and 'metadata' in result:
                metadata = result['metadata']
                val_dist = metadata['class_distribution']['val']
                if 0 in val_dist and 1 in val_dist:
                    ratio = min(val_dist[0], val_dist[1]) / max(val_dist[0], val_dist[1])
                    balance_ratios.append(ratio)
                    dataset_labels.append(result['dataset'].split('/')[-1][:15])
        
        if balance_ratios:
            bars = ax.barh(range(len(balance_ratios)), balance_ratios, 
                          color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(dataset_labels)))
            ax.set_yticklabels(dataset_labels, fontsize=9)
            ax.set_xlabel('Balance Ratio (min/max)', fontsize=11, fontweight='bold')
            ax.set_title('Class Balance Validation (First 20 Datasets)', fontsize=13, fontweight='bold')
            ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Perfect Balance')
            ax.axvline(0.4, color='orange', linestyle='--', linewidth=2, label='Threshold')
            ax.legend()
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'class_balance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Class balance plot saved")
        
        print(f"\n📊 All plots saved to: {plot_dir}/")
    
    def generate_report(self, output_path='results/validation_report.txt'):
        """Generate comprehensive validation report"""
        print("\n" + "="*70)
        print("Generating Validation Report")
        print("="*70)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("DATA VALIDATION REPORT\n")
            f.write("FedED-SegNAS Phase 1 Completion\n")
            f.write("="*70 + "\n\n")
            
            # Summary
            total_datasets = len(self.validation_results)
            passed_datasets = sum(1 for r in self.validation_results if r['passed'])
            failed_datasets = total_datasets - passed_datasets
            
            f.write(f"SUMMARY\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Total Datasets Validated: {total_datasets}\n")
            f.write(f"Passed: {passed_datasets} ({passed_datasets/total_datasets*100:.1f}%)\n")
            f.write(f"Failed: {failed_datasets} ({failed_datasets/total_datasets*100:.1f}%)\n")
            f.write(f"\n")
            
            # Group by model
            models = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0})
            for result in self.validation_results:
                model_name = result['dataset'].split('/')[0]
                models[model_name]['total'] += 1
                if result['passed']:
                    models[model_name]['passed'] += 1
                else:
                    models[model_name]['failed'] += 1
            
            f.write(f"BY MODEL\n")
            f.write(f"{'-'*70}\n")
            for model, stats in sorted(models.items()):
                f.write(f"{model}:\n")
                f.write(f"  Passed: {stats['passed']}/{stats['total']}\n")
                if stats['failed'] > 0:
                    f.write(f"  Failed: {stats['failed']}\n")
                f.write(f"\n")
            
            # Validation checks summary
            f.write(f"VALIDATION CHECKS\n")
            f.write(f"{'-'*70}\n")
            
            check_summary = defaultdict(lambda: {'passed': 0, 'failed': 0})
            for result in self.validation_results:
                if 'checks' in result:
                    for check_name, check_result in result['checks'].items():
                        if check_result['passed']:
                            check_summary[check_name]['passed'] += 1
                        else:
                            check_summary[check_name]['failed'] += 1
            
            for check_name, counts in sorted(check_summary.items()):
                total = counts['passed'] + counts['failed']
                f.write(f"{check_name.replace('_', ' ').title()}:\n")
                f.write(f"  Passed: {counts['passed']}/{total} ({counts['passed']/total*100:.1f}%)\n")
                if counts['failed'] > 0:
                    f.write(f"  Failed: {counts['failed']}\n")
                f.write(f"\n")
            
            # Failed checks details
            if self.failed_checks:
                f.write(f"FAILED CHECKS DETAILS\n")
                f.write(f"{'-'*70}\n")
                for failure in self.failed_checks:
                    f.write(f"  ❌ {failure}\n")
                f.write(f"\n")
            else:
                f.write(f"✅ ALL CHECKS PASSED!\n\n")
            
            # Statistical summary
            f.write(f"STATISTICAL SUMMARY\n")
            f.write(f"{'-'*70}\n")
            
            total_samples = {'train': 0, 'val': 0, 'test': 0}
            total_features = []
            
            for result in self.validation_results:
                if result['passed'] and 'metadata' in result:
                    metadata = result['metadata']
                    total_samples['train'] += metadata['num_train_samples']
                    total_samples['val'] += metadata['num_val_samples']
                    total_samples['test'] += metadata['num_test_samples']
                    total_features.append(metadata['num_features'])
            
            f.write(f"Total Samples Across All Valid Datasets:\n")
            f.write(f"  Training: {total_samples['train']:,}\n")
            f.write(f"  Validation: {total_samples['val']:,}\n")
            f.write(f"  Test: {total_samples['test']:,}\n")
            f.write(f"  Total: {sum(total_samples.values()):,}\n")
            f.write(f"\n")
            
            if total_features:
                f.write(f"Feature Counts:\n")
                f.write(f"  Min: {min(total_features)}\n")
                f.write(f"  Max: {max(total_features)}\n")
                f.write(f"  Unique: {len(set(total_features))}\n")
                f.write(f"\n")
            
            # MAF statistics
            all_maf_means = [r['maf_mean'] for r in self.validation_results 
                            if r['passed'] and 'maf_mean' in r]
            if all_maf_means:
                f.write(f"Minor Allele Frequency (MAF) Statistics:\n")
                f.write(f"  Overall Mean: {np.mean(all_maf_means):.4f}\n")
                f.write(f"  Std Dev: {np.std(all_maf_means):.4f}\n")
                f.write(f"  Min: {min(all_maf_means):.4f}\n")
                f.write(f"  Max: {max(all_maf_means):.4f}\n")
                f.write(f"\n")
            
            # Footer
            f.write("="*70 + "\n")
            f.write(f"Report generated for {total_datasets} datasets\n")
            f.write(f"Processed data location: {self.processed_dir}\n")
            f.write("="*70 + "\n")
        
        print(f"  ✅ Report saved to: {output_path}")
        
        # Print to console
        with open(output_path, 'r') as f:
            print(f"\n{f.read()}")
    
    def run_validation(self):
        """Run validation on all preprocessed datasets"""
        print("="*70)
        print("DATA VALIDATION")
        print("="*70)
        
        # Find all .npz files
        pattern = f"{self.processed_dir}/**/*.npz"
        datasets = sorted(glob.glob(pattern, recursive=True))
        
        print(f"\nFound {len(datasets)} preprocessed datasets")
        print(f"Location: {self.processed_dir}")
        print("="*70)
        
        if len(datasets) == 0:
            print("\n❌ No preprocessed datasets found!")
            return
        
        print("\nValidating datasets...")
        
        # Validate each dataset
        passed = 0
        failed = 0
        
        for dataset_path in datasets:
            result = self.validate_dataset(dataset_path)
            if result['passed']:
                passed += 1
                print(f"  ✅ {result['dataset']}")
            else:
                failed += 1
                print(f"  ❌ {result['dataset']}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
        
        print("\n" + "="*70)
        print(f"Validation Complete: {passed} passed, {failed} failed")
        print("="*70)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        self.generate_report()
        
        return passed, failed


if __name__ == '__main__':
    import os
    os.chdir('/app/FedED-SegNAS')
    
    validator = DataValidator()
    passed, failed = validator.run_validation()
    
    if failed == 0:
        print("\n" + "🎉"*35)
        print("ALL VALIDATION CHECKS PASSED!")
        print("🎉"*35)
        exit(0)
    else:
        print(f"\n⚠️  {failed} datasets failed validation")
        exit(1)
