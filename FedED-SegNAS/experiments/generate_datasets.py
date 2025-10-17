#!/usr/bin/env python3
"""
GAMETES Dataset Generation Script
Generates simulated epistasis datasets for 8 disease models
"""

import subprocess
import os
import yaml
import time
from pathlib import Path
import argparse


class DatasetGenerator:
    def __init__(self, config_path='config/experiment_config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.models = self.dataset_config['models']
        self.snp_sizes = self.dataset_config['simulated']['snp_sizes']
        self.epistasis_orders = self.dataset_config['simulated']['epistasis_orders']
        self.num_datasets = self.dataset_config['simulated']['num_datasets_per_config']
        self.num_cases = self.dataset_config['simulated']['num_cases']
        self.num_controls = self.dataset_config['simulated']['num_controls']
        
        self.gametes_jar = 'data/simulated/GAMETES_2.0.jar'
        
    def generate_single_dataset(self, model, order, num_snps, dataset_id):
        """
        Generate a single dataset using GAMETES
        
        Args:
            model: Dictionary with model configuration
            order: Epistasis order (2 or 3)
            num_snps: Total number of SNPs
            dataset_id: Dataset identifier (for random seed)
        """
        model_name = model['name']
        output_dir = f"data/simulated/{model_name}/order{order}/snps{num_snps}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/dataset_{dataset_id}"
        
        # Build GAMETES command
        # GAMETES uses -M flag for model generation
        cmd = [
            'java', '-jar', self.gametes_jar,
            '-D',
            f'-a {num_snps}',  # Total attributes/SNPs
            f'-s {self.num_cases}',  # Cases
            f'-w {self.num_controls}',  # Controls
            f'-r 1',  # One replicate
            f'-o {output_file}'
        ]
        
        # Build model specification
        model_spec = [
            '-M',
            f'-h {model["heritability"]}',  # Heritability
            f'-a {model["maf"]}',  # MAF
            f'-o {output_file}_model.txt'  # Model output
        ]
        
        # Combine commands
        full_cmd = ' '.join(cmd + model_spec)
        
        try:
            # Run GAMETES
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, f"Generated: {output_file}"
            else:
                return False, f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout generating {output_file}"
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def generate_all_datasets(self, test_mode=False):
        """
        Generate all datasets according to configuration
        
        Args:
            test_mode: If True, generate only 2 datasets per config for testing
        """
        print("="*70)
        print("GAMETES Dataset Generation")
        print("="*70)
        
        num_datasets_to_generate = 2 if test_mode else self.num_datasets
        
        total_datasets = (
            len(self.models) * 
            len(self.epistasis_orders) * 
            len(self.snp_sizes) * 
            num_datasets_to_generate
        )
        
        print(f"\nConfiguration:")
        print(f"  Models: {len(self.models)}")
        print(f"  Epistasis orders: {self.epistasis_orders}")
        print(f"  SNP sizes: {self.snp_sizes}")
        print(f"  Datasets per config: {num_datasets_to_generate}")
        print(f"  Total datasets to generate: {total_datasets}")
        print(f"  Test mode: {test_mode}")
        print("="*70)
        
        generated_count = 0
        failed_count = 0
        start_time = time.time()
        
        for model in self.models:
            print(f"\n[Model: {model['name']}]")
            print(f"  Heritability: {model['heritability']}, MAF: {model['maf']}")
            print(f"  Type: {model['type']}, Marginal: {model['marginal']}")
            
            for order in self.epistasis_orders:
                print(f"\n  [Epistasis Order: {order}]")
                
                for num_snps in self.snp_sizes:
                    print(f"    [SNPs: {num_snps}] ", end='', flush=True)
                    
                    success_batch = 0
                    for dataset_id in range(num_datasets_to_generate):
                        success, message = self.generate_single_dataset(
                            model, order, num_snps, dataset_id
                        )
                        
                        if success:
                            generated_count += 1
                            success_batch += 1
                        else:
                            failed_count += 1
                            if failed_count < 5:  # Only print first few errors
                                print(f"\n      Failed: {message}")
                        
                        # Progress indicator
                        if (dataset_id + 1) % 10 == 0:
                            print(f".", end='', flush=True)
                    
                    print(f" ✓ ({success_batch}/{num_datasets_to_generate})")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("Generation Complete!")
        print("="*70)
        print(f"Successfully generated: {generated_count}/{total_datasets} datasets")
        print(f"Failed: {failed_count}")
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"Average time per dataset: {elapsed_time/max(generated_count,1):.2f} seconds")
        print("="*70)
        
        return generated_count, failed_count


def main():
    parser = argparse.ArgumentParser(description='Generate GAMETES datasets')
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test mode: generate only 2 datasets per configuration'
    )
    parser.add_argument(
        '--config',
        default='config/experiment_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir('/app/FedED-SegNAS')
    
    # Initialize generator
    generator = DatasetGenerator(args.config)
    
    # Generate datasets
    generated, failed = generator.generate_all_datasets(test_mode=args.test)
    
    if generated > 0:
        print(f"\n✅ Dataset generation completed successfully!")
        print(f"📁 Datasets saved in: data/simulated/")
    else:
        print(f"\n❌ Dataset generation failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
