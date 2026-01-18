#!/usr/bin/env python3
"""
Unit tests for PSO-based Neural Architecture Search

Tests all NAS components:
- SearchSpace class
- Particle class
- FitnessCalculator class
- MultiObjectivePSO class
- Integration with federated learning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from NAS.search_space import SearchSpace
from NAS.pso_nas import Particle, FitnessCalculator, MultiObjectivePSO


def test_search_space():
    """Test SearchSpace class"""
    print("\n" + "="*70)
    print("TEST 1: SearchSpace")
    print("="*70)
    
    space = SearchSpace()
    
    # Test random architecture generation
    arch = space.random_architecture()
    assert 'num_blocks' in arch, "Architecture missing 'num_blocks'"
    assert 'blocks' in arch, "Architecture missing 'blocks'"
    assert 'dense_1' in arch, "Architecture missing 'dense_1'"
    assert 'dense_2' in arch, "Architecture missing 'dense_2'"
    assert 'dropout' in arch, "Architecture missing 'dropout'"
    assert len(arch['blocks']) == arch['num_blocks'], "Block count mismatch"
    
    # Test encoding
    position = space.encode_architecture(arch)
    assert len(position) == 16, f"Position size should be 16, got {len(position)}"
    
    # Test decoding
    decoded = space.decode_position(position)
    assert decoded['num_blocks'] == arch['num_blocks'], "Decoded num_blocks mismatch"
    assert len(decoded['blocks']) == arch['num_blocks'], "Decoded blocks count mismatch"
    
    # Test valid ranges
    assert 1 <= arch['num_blocks'] <= 4, "num_blocks out of range"
    for block in arch['blocks']:
        assert 32 <= block['filters'] <= 256, "filters out of range"
        assert 3 <= block['kernel_size'] <= 7, "kernel_size out of range"
        assert 0 <= block['pool_type'] <= 2, "pool_type out of range"
    assert 128 <= arch['dense_1'] <= 512, "dense_1 out of range"
    assert 128 <= arch['dense_2'] <= 512, "dense_2 out of range"
    assert 0.2 <= arch['dropout'] <= 0.5, "dropout out of range"
    
    print(">> SearchSpace test passed [OK]")
    print("   - Random architecture generation works")
    print("   - Encoding/decoding works")
    print("   - All ranges validated")


def test_particle():
    """Test Particle class"""
    print("\n" + "="*70)
    print("TEST 2: Particle")
    print("="*70)
    
    space = SearchSpace()
    arch = space.random_architecture()
    position = space.encode_architecture(arch)
    velocity = np.random.randn(len(position)) * 0.1
    
    particle = Particle(position, velocity)
    
    # Test initialization
    assert particle.fitness == float('inf'), "Initial fitness should be inf"
    assert particle.pbest_fitness == float('inf'), "Initial pbest_fitness should be inf"
    assert np.array_equal(particle.position, position), "Position not initialized correctly"
    assert np.array_equal(particle.velocity, velocity), "Velocity not initialized correctly"
    
    # Test velocity update
    gbest = position + np.random.randn(len(position))
    old_velocity = particle.velocity.copy()
    particle.update_velocity(gbest)
    assert not np.array_equal(particle.velocity, old_velocity), "Velocity should change"
    
    # Test position update
    old_position = particle.position.copy()
    particle.update_position(space)
    assert not np.array_equal(particle.position, old_position), "Position should change"
    
    # Test position clipping
    assert 1 <= particle.position[0] <= 4, "num_blocks not clipped correctly"
    
    # Test pbest update
    particle.fitness = 0.5
    updated = particle.update_pbest()
    assert updated == True, "Pbest should update when fitness improves"
    assert particle.pbest_fitness == 0.5, "Pbest fitness not updated"
    assert np.array_equal(particle.pbest_position, particle.position), "Pbest position not updated"
    
    # Test pbest doesn't update when fitness is worse
    particle.fitness = 0.6
    updated = particle.update_pbest()
    assert updated == False, "Pbest should not update when fitness is worse"
    assert particle.pbest_fitness == 0.5, "Pbest fitness should remain 0.5"
    
    print(">> Particle test passed [OK]")
    print("   - Initialization works")
    print("   - Velocity update works (PSO Equation 21)")
    print("   - Position update works (PSO Equation 22)")
    print("   - Pbest tracking works")


def test_fitness_calculator():
    """Test FitnessCalculator class"""
    print("\n" + "="*70)
    print("TEST 3: FitnessCalculator")
    print("="*70)
    
    space = SearchSpace()
    arch = space.random_architecture()
    
    calculator = FitnessCalculator(num_snps=50)
    
    # Test f2 (blocks)
    f2 = calculator.calculate_f2_blocks(arch)
    expected_f2 = arch['num_blocks'] / 4.0  # Normalized
    assert f2 == expected_f2, f"f2 should equal num_blocks/4, got {f2}"
    assert isinstance(f2, float), "f2 should be float"
    assert 0 <= f2 <= 1.0, "f2 should be in [0, 1]"
    
    # Test f3 (parameters)
    f3 = calculator.calculate_f3_parameters(arch)
    assert 0 <= f3 <= 1.0, f"f3 should be in [0, 1], got {f3}"
    assert isinstance(f3, float), "f3 should be float"
    
    # Test f4 (GFLOPs)
    f4 = calculator.calculate_f4_gflops(arch)
    assert f4 >= 0, f"f4 should be non-negative, got {f4}"
    assert isinstance(f4, float), "f4 should be float"
    
    # Test model building
    model = calculator.build_model_from_architecture(arch)
    assert model is not None, "Model should be built"
    assert model.count_params() > 0, "Model should have parameters"
    
    print(">> FitnessCalculator test passed [OK]")
    print(f"   - f2 (blocks): {f2}")
    print(f"   - f3 (parameters): {f3:.4f}")
    print(f"   - f4 (GFLOPs): {f4:.6f}")
    print("   - Model building works")


def test_pso_optimizer():
    """Test MultiObjectivePSO class"""
    print("\n" + "="*70)
    print("TEST 4: MultiObjectivePSO")
    print("="*70)
    
    # Create small dummy data
    X_train = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    y_train = np.random.randint(0, 2, 50).astype(np.int32)
    X_val = np.random.randint(0, 3, (30, 50)).astype(np.float32)
    y_val = np.random.randint(0, 2, 30).astype(np.int32)
    
    # Initialize PSO with small parameters
    pso = MultiObjectivePSO(
        num_snps=50,
        num_particles=2,
        max_iterations=2
    )
    
    # Test initialization
    assert len(pso.particles) == 2, "Should have 2 particles"
    assert pso.gbest_fitness == float('inf'), "Initial gbest_fitness should be inf"
    assert pso.gbest_position is not None, "gbest_position should be initialized"
    
    # Test stage determination
    assert pso.get_stage(5, 30) == 1, "Round 5/30 should be stage 1"
    assert pso.get_stage(15, 30) == 2, "Round 15/30 should be stage 2"
    assert pso.get_stage(25, 30) == 3, "Round 25/30 should be stage 3"
    
    # Test search (quick)
    print("   Running PSO search (2 particles, 2 iterations)...")
    best_arch, best_fitness = pso.search(
        X_train, y_train, X_val, y_val,
        current_round=1, max_rounds=30,
        verbose=False
    )
    
    assert best_arch is not None, "Best architecture should be found"
    assert best_fitness < float('inf'), "Best fitness should be finite"
    assert 'num_blocks' in best_arch, "Best architecture should have num_blocks"
    assert 'blocks' in best_arch, "Best architecture should have blocks"
    
    # Test that gbest was updated
    assert pso.gbest_fitness < float('inf'), "Gbest fitness should be updated"
    
    print(">> MultiObjectivePSO test passed [OK]")
    print(f"   - Initialization works")
    print(f"   - Stage determination works")
    print(f"   - PSO search works")
    print(f"   - Best fitness: {best_fitness:.4f}")
    print(f"   - Best architecture: {best_arch['num_blocks']} blocks")


def test_integration():
    """Test full NAS integration with federated learning"""
    print("\n" + "="*70)
    print("TEST 5: Full Integration")
    print("="*70)
    
    from Phase3.federated_learning import SimpleFederatedTrainer
    
    # Create dummy federated data
    federated_data = {}
    for i in range(3):
        federated_data[f'client_{i}_X'] = np.random.randint(0, 3, (50, 50)).astype(np.float32)
        federated_data[f'client_{i}_y'] = np.random.randint(0, 2, 50).astype(np.int32)
    
    federated_data['validation_X'] = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    federated_data['validation_y'] = np.random.randint(0, 2, 50).astype(np.int32)
    federated_data['test_X'] = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    federated_data['test_y'] = np.random.randint(0, 2, 50).astype(np.int32)
    
    # Test with NAS
    print("   Initializing federated trainer with NAS...")
    trainer = SimpleFederatedTrainer(num_snps=50, num_clients=3, use_nas=True)
    
    print("   Running federated training (6 rounds, NAS at round 5)...")
    history = trainer.train(
        federated_data,
        num_rounds=6,
        clients_per_round=2,
        local_epochs=1,
        nas_frequency=5,
        verbose=False
    )
    
    # Check NAS was run
    assert 'nas_searches' in history, "History should contain nas_searches"
    assert len(history['nas_searches']) >= 1, "At least one NAS search should have run"
    
    # Check NAS search structure
    nas_search = history['nas_searches'][0]
    assert 'round' in nas_search, "NAS search should have round"
    assert 'architecture' in nas_search, "NAS search should have architecture"
    assert 'fitness' in nas_search, "NAS search should have fitness"
    
    # Check architecture structure
    arch = nas_search['architecture']
    assert 'num_blocks' in arch, "Architecture should have num_blocks"
    assert 'blocks' in arch, "Architecture should have blocks"
    assert 'dense_1' in arch, "Architecture should have dense_1"
    
    # Check training continued after NAS
    assert len(history['rounds']) == 6, "Should have 6 rounds"
    assert len(history['val_accuracy']) == 6, "Should have 6 validation accuracies"
    
    print(">> Full integration test passed [OK]")
    print(f"   - Federated trainer with NAS works")
    print(f"   - NAS searches: {len(history['nas_searches'])}")
    print(f"   - NAS ran at round: {nas_search['round']}")
    print(f"   - Architecture found: {arch['num_blocks']} blocks")
    print(f"   - Training continued after NAS")


def test_segmented_strategy():
    """Test segmented fitness strategy"""
    print("\n" + "="*70)
    print("TEST 6: Segmented Strategy")
    print("="*70)
    
    space = SearchSpace()
    arch = space.random_architecture()
    calculator = FitnessCalculator(num_snps=50)
    
    # Create dummy data
    X_train = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    y_train = np.random.randint(0, 2, 50).astype(np.int32)
    X_val = np.random.randint(0, 3, (30, 50)).astype(np.float32)
    y_val = np.random.randint(0, 2, 30).astype(np.int32)
    
    # Test Stage 1 (all objectives)
    fitness_stage1 = calculator.calculate_fitness(
        arch, X_train, y_train, X_val, y_val,
        stage=1, current_round=5, max_rounds=50
    )
    assert fitness_stage1 > 0, "Stage 1 fitness should be positive"
    
    # Test Stage 2 (accuracy + GFLOPs)
    fitness_stage2 = calculator.calculate_fitness(
        arch, X_train, y_train, X_val, y_val,
        stage=2, current_round=20, max_rounds=50
    )
    assert fitness_stage2 > 0, "Stage 2 fitness should be positive"
    
    # Test Stage 3 (accuracy + parameters)
    fitness_stage3 = calculator.calculate_fitness(
        arch, X_train, y_train, X_val, y_val,
        stage=3, current_round=40, max_rounds=50
    )
    assert fitness_stage3 > 0, "Stage 3 fitness should be positive"
    
    print(">> Segmented strategy test passed [OK]")
    print(f"   - Stage 1 fitness: {fitness_stage1:.4f}")
    print(f"   - Stage 2 fitness: {fitness_stage2:.4f}")
    print(f"   - Stage 3 fitness: {fitness_stage3:.4f}")
    print("   - All stages work correctly")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("PSO-NAS COMPREHENSIVE UNIT TESTS")
    print("="*80)
    print("\nTesting all NAS components...")
    
    try:
        test_search_space()
        test_particle()
        test_fitness_calculator()
        test_pso_optimizer()
        test_integration()
        test_segmented_strategy()
        
        print("\n" + "="*80)
        print("[OK] ALL TESTS PASSED (6/6)")
        print("="*80)
        print("\nTest Summary:")
        print("  [OK] SearchSpace: Architecture encoding/decoding")
        print("  [OK] Particle: PSO velocity/position updates")
        print("  [OK] FitnessCalculator: Multi-objective fitness")
        print("  [OK] MultiObjectivePSO: Complete PSO search")
        print("  [OK] Integration: FL + NAS end-to-end")
        print("  [OK] Segmented Strategy: 3-stage fitness")
        print("\n" + "="*80)
        print("NAS IMPLEMENTATION VERIFIED [OK]")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("[FAIL] TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("="*80 + "\n")
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*80)
        print("[FAIL] TEST ERROR")
        print("="*80)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        sys.exit(1)
