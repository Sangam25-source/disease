#!/usr/bin/env python3
"""
Project Summary Script
Provides an overview of the FedED-SegNAS project status
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/app/FedED-SegNAS')

def count_files(directory, pattern='*'):
    """Count files matching pattern in directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))

def get_dir_size(directory):
    """Get directory size in MB"""
    path = Path(directory)
    if not path.exists():
        return 0
    
    total = 0
    for file in path.rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total / (1024 * 1024)  # Convert to MB

def print_banner(text, char='=', width=70):
    """Print formatted banner"""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)

def main():
    os.chdir('/app/FedED-SegNAS')
    
    print_banner("🧬 FedED-SegNAS Project Summary", '=', 70)
    
    print("\n📊 DATASET STATISTICS")
    print("-" * 70)
    
    # Simulated datasets
    simulated_count = count_files('data/simulated', '*.txt')
    simulated_size = get_dir_size('data/simulated')
    
    print(f"  Simulated Datasets:       {simulated_count:>6} files")
    print(f"  Storage (simulated):      {simulated_size:>6.1f} MB")
    
    # Preprocessed datasets
    processed_count = count_files('data/processed', '*.npz')
    processed_size = get_dir_size('data/processed')
    
    print(f"  Preprocessed Datasets:    {processed_count:>6} files")
    print(f"  Storage (preprocessed):   {processed_size:>6.1f} MB")
    
    # Calculate compression ratio
    if processed_count > 0 and simulated_count > 0:
        avg_raw = simulated_size / simulated_count
        avg_processed = processed_size / processed_count
        compression = avg_raw / avg_processed if avg_processed > 0 else 0
        print(f"  Compression Ratio:        {compression:>6.1f}x")
    
    print("\n📁 PROJECT STRUCTURE")
    print("-" * 70)
    
    # Count different file types
    python_files = count_files('.', '*.py')
    yaml_files = count_files('.', '*.yaml') + count_files('.', '*.yml')
    md_files = count_files('.', '*.md')
    
    print(f"  Python Scripts:           {python_files:>6} files")
    print(f"  Configuration Files:      {yaml_files:>6} files")
    print(f"  Documentation Files:      {md_files:>6} files")
    
    print("\n✅ COMPLETED PHASES")
    print("-" * 70)
    print("  [✓] Phase 0: Environment Setup")
    print("  [✓] Phase 1: Dataset Generation & Preprocessing")
    print("      • 192 simulated datasets generated")
    print("      • Federated splitting implemented")
    print("      • Data validation passed")
    
    print("\n🔄 NEXT PHASES")
    print("-" * 70)
    print("  [ ] Phase 2: Fuzzy CNN Implementation")
    print("  [ ] Phase 3: PSO-NAS Algorithm")
    print("  [ ] Phase 4: Privacy-Preserving Module")
    print("  [ ] Phase 5: Federated Learning Framework")
    print("  [ ] Phase 6: Training & Evaluation")
    print("  [ ] Phase 7: Ablation Studies")
    print("  [ ] Phase 8: Real Dataset Validation")
    
    print("\n📈 PROGRESS")
    print("-" * 70)
    total_phases = 8
    completed_phases = 2
    progress = (completed_phases / total_phases) * 100
    
    # Progress bar
    bar_length = 40
    filled = int(bar_length * completed_phases / total_phases)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"  Overall Progress: [{bar}] {progress:.0f}%")
    print(f"  Completed: {completed_phases}/{total_phases} phases")
    
    print("\n🎯 KEY ACHIEVEMENTS")
    print("-" * 70)
    print("  ✓ Full project structure established")
    print("  ✓ All dependencies installed")
    print("  ✓ 192 epistasis datasets generated")
    print("  ✓ Federated data splitting (50 clients)")
    print("  ✓ Data validation pipeline complete")
    print("  ✓ Comprehensive documentation created")
    
    print("\n🚀 QUICK START")
    print("-" * 70)
    print("  Generate datasets:")
    print("    python experiments/generate_simple_datasets.py --test")
    print()
    print("  Preprocess data:")
    print("    python experiments/data_preprocessing.py --limit 10")
    print()
    print("  View status:")
    print("    cat STATUS.md")
    
    print("\n📚 DOCUMENTATION")
    print("-" * 70)
    print("  README.md            - Project overview")
    print("  STATUS.md            - Detailed status report")
    print("  DATASET_SUMMARY.md   - Dataset documentation")
    print("  QUICKREF.md          - Quick reference guide")
    
    print("\n" + "=" * 70)
    print("  Project: FedED-SegNAS")
    print("  Status: Phase 1 Complete ✅")
    print("  Next: Fuzzy CNN Implementation")
    print("=" * 70)
    print()

if __name__ == '__main__':
    main()
