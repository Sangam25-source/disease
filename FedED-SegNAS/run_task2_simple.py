#!/usr/bin/env python3
"""
TASK 2 SIMPLE RUNNER: Increase Training Rounds Dramatically
===========================================================

Simple script to run Task 2 using the existing train_federated.py script.
This will train all available datasets with increased rounds.

Usage:
------
# Full training (recommended)
python run_task2_simple.py

# Quick test (reduced rounds)
python run_task2_simple.py --quick

Author: FedED-SegNAS Project
Date: January 2026
"""

import sys
import os
import subprocess
import argparse
import glob
from pathlib import Path


def find_all_datasets():
    """Find all available datasets"""
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
        
        # Find all dataset files
        for dataset_file in model_dir.rglob('dataset_*.npz'):
            # Extract SNP count from path
            snp_count = None
            for part in dataset_file.parts:
                if part.startswith('snps'):
                    snp_count = int(part.replace('snps', ''))
                    break
            
            if snp_count:
                datasets.append((model_name, snp_count, str(dataset_file)))
    
    return sorted(datasets)


def run_training_command(model_name, snp_count, rounds, clients_per_round, local_epochs, quick=False):
    """Run training command for a specific model and SNP count"""
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter (with venv)
        'Phase 3/train_federated.py',
        '--model', model_name,
        '--snps', str(snp_count),
        '--rounds', str(rounds),
        '--clients-per-round', str(clients_per_round),
        '--local-epochs', str(local_epochs)
    ]
    
    if quick:
        cmd.append('--quick')
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name} with {snp_count} SNPs")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True, cwd='.')
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {model_name} with {snp_count} SNPs completed")
            return True
        else:
            print(f"❌ FAILED: {model_name} with {snp_count} SNPs failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {model_name} with {snp_count} SNPs - {e}")
        return False


def main():
    """Main function to run Task 2 training"""
    parser = argparse.ArgumentParser(
        description='Task 2: Increase Training Rounds Dramatically (Simple Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TASK 2 IMPLEMENTATION: Increase Training Rounds Dramatically
===========================================================

This implements Task 2 from the accuracy improvement roadmap using
the existing train_federated.py script with enhanced parameters:

✅ INCREASED ROUNDS: 100 minimum (vs 10 previously)
✅ INCREASED CLIENTS: 15 per round (vs 10 previously)  
✅ INCREASED LOCAL EPOCHS: 10 (vs 5 previously)
✅ AUTOMATIC TRAINING: All models and datasets

Expected Improvements:
- +5-10% accuracy from increased training rounds
- Better convergence from more training

Time Required: 2-4 hours for full training
        """
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (reduced rounds for testing)'
    )
    
    parser.add_argument(
        '--rounds', type=int, default=100,
        help='Number of training rounds (default: 100)'
    )
    
    parser.add_argument(
        '--clients-per-round', type=int, default=15,
        help='Number of clients per round (default: 15)'
    )
    
    parser.add_argument(
        '--local-epochs', type=int, default=10,
        help='Number of local epochs (default: 10)'
    )
    
    parser.add_argument(
        '--models', type=str, nargs='+', 
        help='Specific models to train (e.g., model1 model2)'
    )
    
    parser.add_argument(
        '--snps', type=int, nargs='+',
        help='Specific SNP counts to train (e.g., 50 100 500)'
    )
    
    args = parser.parse_args()
    
    # Adjust for quick test
    if args.quick:
        args.rounds = 20
        print("🚀 QUICK TEST MODE ENABLED")
        print("   Reduced rounds for testing purposes")
    
    print("\n" + "="*80)
    print("TASK 2: INCREASE TRAINING ROUNDS DRAMATICALLY (SIMPLE)")
    print("="*80)
    print("Implementation Features:")
    print(f"  ✅ Increased rounds: {args.rounds} (vs 10 previously)")
    print(f"  ✅ Increased clients per round: {args.clients_per_round} (vs 10 previously)")
    print(f"  ✅ Increased local epochs: {args.local_epochs} (vs 5 previously)")
    print("  ✅ Automatic training: all models and datasets")
    print("\nExpected Improvements:")
    print("  📈 +5-10% accuracy from increased training rounds")
    print("  📈 Better convergence from more training")
    
    if not args.quick:
        print(f"\n⏰ ESTIMATED TIME: 2-4 hours for full training")
        print("   (20-40 minutes per model/SNP combination)")
        
        response = input("\nProceed with full training? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled. Use --quick for testing.")
            return 1
    
    print("="*80 + "\n")
    
    # Check if train_federated.py exists
    train_script = Path('Phase 3/train_federated.py')
    if not train_script.exists():
        print(f"❌ ERROR: Training script not found: {train_script}")
        print("   Please ensure you're running from the FedED-SegNAS directory")
        return 1
    
    # Find all datasets
    print("🔍 Scanning for available datasets...")
    datasets = find_all_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        print("   Please ensure data is preprocessed in data/processed/")
        return 1
    
    print(f"✅ Found {len(datasets)} dataset combinations")
    
    # Filter datasets if specific models/SNPs requested
    if args.models:
        datasets = [(m, s, p) for m, s, p in datasets if m in args.models]
        print(f"   Filtered to {len(datasets)} datasets for models: {args.models}")
    
    if args.snps:
        datasets = [(m, s, p) for m, s, p in datasets if s in args.snps]
        print(f"   Filtered to {len(datasets)} datasets for SNPs: {args.snps}")
    
    # Group by model and SNP count (we'll train each combination once)
    unique_combinations = list(set((model, snps) for model, snps, _ in datasets))
    unique_combinations.sort()
    
    print(f"\n📊 Will train {len(unique_combinations)} model/SNP combinations:")
    for model, snps in unique_combinations:
        print(f"   {model} with {snps} SNPs")
    
    print(f"\n🚀 Starting training...")
    
    # Train each combination
    successful = 0
    failed = 0
    
    for i, (model_name, snp_count) in enumerate(unique_combinations, 1):
        print(f"\n📊 Progress: {i}/{len(unique_combinations)} combinations")
        
        success = run_training_command(
            model_name, snp_count, 
            args.rounds, args.clients_per_round, args.local_epochs,
            args.quick
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"TASK 2 TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total combinations: {len(unique_combinations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(unique_combinations)*100:.1f}%")
    
    if successful > 0:
        print(f"\n🎉 TASK 2 IMPLEMENTATION COMPLETE!")
        print(f"   Successfully trained {successful} model/SNP combinations")
        print(f"   Results saved in: results/federated_training/")
        print(f"\n📊 Check results with:")
        print(f"   ls -la results/federated_training/")
        
        return 0
    else:
        print(f"\n❌ No training completed successfully")
        return 1


if __name__ == '__main__':
    sys.exit(main())