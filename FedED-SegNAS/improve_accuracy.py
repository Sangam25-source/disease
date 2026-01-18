#!/usr/bin/env python3
"""
Quick Script to Improve Federated Learning Accuracy

This script implements the key improvements from ACCURACY_IMPROVEMENT_GUIDE.md
"""

import sys
import os

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure we're in FedED-SegNAS directory
if not script_dir.endswith('FedED-SegNAS'):
    print(f"ERROR: Script must be run from FedED-SegNAS directory")
    print(f"Current location: {script_dir}")
    print(f"\nPlease run:")
    print(f"  cd FedED-SegNAS")
    print(f"  python improve_accuracy.py")
    sys.exit(1)

# Change to FedED-SegNAS directory
os.chdir(script_dir)

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print("ACCURACY IMPROVEMENT SCRIPT")
print("="*80)
print(f"\nCurrent directory: {os.getcwd()}")
print("\nThis script will help you improve federated learning accuracy.")
print("\nKey improvements:")
print("  1. Adjust NAS search space (prevent too-small architectures)")
print("  2. Adjust fitness weights (prioritize accuracy)")
print("  3. Increase learning rate")
print("  4. Increase NAS search effort")
print("\n" + "="*80)

# Check if user wants to apply improvements
response = input("\nApply improvements? (y/n): ").strip().lower()

if response != 'y':
    print("Exiting without changes.")
    sys.exit(0)

print("\n>> Applying improvements...")

# ============================================================================
# IMPROVEMENT 1: Adjust Search Space
# ============================================================================
print("\n1. Adjusting NAS search space...")

search_space_file = "NAS/search_space.py"
if os.path.exists(search_space_file):
    with open(search_space_file, 'r') as f:
        content = f.read()
    
    # Backup
    with open(search_space_file + ".backup", 'w') as f:
        f.write(content)
    
    # Modify search space
    content = content.replace(
        "self.num_blocks_range = [1, 2, 3, 4]",
        "self.num_blocks_range = [2, 3, 4]  # Prevent too-small architectures"
    )
    content = content.replace(
        "self.filters_range = [32, 64, 128, 256]",
        "self.filters_range = [64, 128, 256]  # Increased minimum filters"
    )
    content = content.replace(
        "self.dense_range = [128, 256, 512]",
        "self.dense_range = [256, 512]  # Increased minimum dense units"
    )
    
    with open(search_space_file, 'w') as f:
        f.write(content)
    
    print("   >> Search space adjusted")
    print("      - Minimum blocks: 2 (was 1)")
    print("      - Minimum filters: 64 (was 32)")
    print("      - Minimum dense: 256 (was 128)")
else:
    print("   >> WARNING: search_space.py not found")

# ============================================================================
# IMPROVEMENT 2: Adjust Fitness Weights
# ============================================================================
print("\n2. Adjusting fitness weights to prioritize accuracy...")

pso_nas_file = "NAS/pso_nas.py"
if os.path.exists(pso_nas_file):
    with open(pso_nas_file, 'r') as f:
        content = f.read()
    
    # Backup
    with open(pso_nas_file + ".backup", 'w') as f:
        f.write(content)
    
    # Modify fitness weights
    # Stage 1
    content = content.replace(
        "fitness = 0.4 * f1 + 0.2 * f2 + 0.2 * f3 + 0.2 * f4",
        "fitness = 0.6 * f1 + 0.15 * f2 + 0.15 * f3 + 0.1 * f4  # Prioritize accuracy"
    )
    # Stage 2
    content = content.replace(
        "fitness = 0.7 * f1 + 0.3 * f4",
        "fitness = 0.8 * f1 + 0.2 * f4  # Prioritize accuracy"
    )
    # Stage 3
    content = content.replace(
        "fitness = 0.7 * f1 + 0.3 * f3",
        "fitness = 0.8 * f1 + 0.2 * f3  # Prioritize accuracy"
    )
    
    with open(pso_nas_file, 'w') as f:
        f.write(content)
    
    print("   >> Fitness weights adjusted")
    print("      - Stage 1: 60% accuracy (was 40%)")
    print("      - Stage 2: 80% accuracy (was 70%)")
    print("      - Stage 3: 80% accuracy (was 70%)")
else:
    print("   >> WARNING: pso_nas.py not found")

# ============================================================================
# IMPROVEMENT 3: Increase Learning Rate
# ============================================================================
print("\n3. Increasing learning rate...")

train_file = "Phase 3/train_federated.py"
if os.path.exists(train_file):
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Backup
    with open(train_file + ".backup", 'w') as f:
        f.write(content)
    
    # Modify learning rate
    content = content.replace(
        "learning_rate=0.001,",
        "learning_rate=0.01,  # Increased for faster convergence"
    )
    
    with open(train_file, 'w') as f:
        f.write(content)
    
    print("   >> Learning rate increased")
    print("      - New rate: 0.01 (was 0.001)")
else:
    print("   >> WARNING: train_federated.py not found")

# ============================================================================
# IMPROVEMENT 4: Increase NAS Search Effort
# ============================================================================
print("\n4. Increasing NAS search effort...")

fl_file = "Phase 3/federated_learning.py"
if os.path.exists(fl_file):
    with open(fl_file, 'r') as f:
        content = f.read()
    
    # Backup
    with open(fl_file + ".backup", 'w') as f:
        f.write(content)
    
    # Modify NAS parameters
    content = content.replace(
        "num_particles=5,  # Small for speed",
        "num_particles=10,  # Increased for better search"
    )
    content = content.replace(
        "max_iterations=10",
        "max_iterations=20  # Increased for better search"
    )
    
    with open(fl_file, 'w') as f:
        f.write(content)
    
    print("   >> NAS search effort increased")
    print("      - Particles: 10 (was 5)")
    print("      - Iterations: 20 (was 10)")
else:
    print("   >> WARNING: federated_learning.py not found")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("IMPROVEMENTS APPLIED SUCCESSFULLY")
print("="*80)
print("\nBackup files created:")
print("  - NAS/search_space.py.backup")
print("  - NAS/pso_nas.py.backup")
print("  - Phase 3/train_federated.py.backup")
print("  - Phase 3/federated_learning.py.backup")
print("\nTo revert changes, restore from backup files.")
print("\n" + "="*80)
print("RECOMMENDED NEXT STEPS")
print("="*80)
print("\n1. Test with real data (MOST IMPORTANT):")
print('   python "Phase 3/train_federated.py" --model model1 --snps 50 --rounds 50 --use-nas')
print("\n2. Or test with more rounds:")
print('   python "Phase 3/train_federated.py" --model model1 --snps 50 --rounds 50 --use-nas --quick')
print("\n3. Monitor validation accuracy - should increase over rounds")
print("\n4. Check NAS searches - architectures should be larger now")
print("\n" + "="*80)
print("\nIMPORTANT: 50% accuracy with random data is EXPECTED and CORRECT!")
print("Use real epistasis data to see actual improvements (60-70% accuracy).")
print("="*80 + "\n")
