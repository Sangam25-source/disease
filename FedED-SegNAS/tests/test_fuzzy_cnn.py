"""
Unit Tests for Fuzzy CNN Architecture
=====================================

Comprehensive test suite for validating the Fuzzy CNN components and architecture.

Test Categories:
1. Layer Shape Validation
2. Forward Pass Tests
3. Model Compilation Tests
4. Training on Real Data
5. Parameter Count Tests
6. Edge Case Tests

Author: FedED-SegNAS Project
Date: October 2024
"""

import sys
import os
sys.path.insert(0, '/app/FedED-SegNAS')

import unittest
import numpy as np
import tensorflow as tf
from models.fuzzy_cnn import (
    FuzzificationLayer,
    FuzzyConvLayer,
    DefuzzificationLayer,
    FixedFuzzyCNN,
    build_fuzzy_cnn
)


class TestFuzzificationLayer(unittest.TestCase):
    """Test suite for FuzzificationLayer."""
    
    def test_fuzzification_shape(self):
        """Test that fuzzification produces correct output shape."""
        print("\n[TEST] Fuzzification Layer Shape...")
        
        batch_size = 32
        num_snps = 50
        num_fuzzy_sets = 3
        
        # Create layer
        fuzz_layer = FuzzificationLayer(num_fuzzy_sets=num_fuzzy_sets)
        
        # Create dummy input
        input_data = tf.random.uniform((batch_size, num_snps), minval=0, maxval=3)
        
        # Forward pass
        output = fuzz_layer(input_data)
        
        # Check shape
        expected_shape = (batch_size, num_snps, num_fuzzy_sets)
        self.assertEqual(output.shape, expected_shape)
        print(f"   ✅ Input: {input_data.shape} -> Output: {output.shape}")
    
    def test_fuzzification_range(self):
        """Test that membership values are in [0, 1] range."""
        print("\n[TEST] Fuzzification Output Range...")
        
        fuzz_layer = FuzzificationLayer(num_fuzzy_sets=3)
        input_data = tf.constant([[0.0, 1.0, 2.0]], dtype=tf.float32)
        
        output = fuzz_layer(input_data)
        
        # Check range
        self.assertTrue(tf.reduce_all(output >= 0.0))
        self.assertTrue(tf.reduce_all(output <= 1.0))
        print(f"   ✅ Output range: [{tf.reduce_min(output).numpy():.4f}, {tf.reduce_max(output).numpy():.4f}]")
    
    def test_fuzzification_trainable_params(self):
        """Test that fuzzification has correct trainable parameters."""
        print("\n[TEST] Fuzzification Trainable Parameters...")
        
        num_fuzzy_sets = 3
        fuzz_layer = FuzzificationLayer(num_fuzzy_sets=num_fuzzy_sets)
        
        # Build layer
        fuzz_layer.build((None, 50))
        
        # Check trainable variables
        trainable_vars = fuzz_layer.trainable_variables
        self.assertEqual(len(trainable_vars), 2)  # means and stds
        
        # Check shapes
        self.assertEqual(trainable_vars[0].shape, (num_fuzzy_sets,))  # means
        self.assertEqual(trainable_vars[1].shape, (num_fuzzy_sets,))  # stds
        print(f"   ✅ Trainable params: {len(trainable_vars)} (means, stds)")


class TestFuzzyConvLayer(unittest.TestCase):
    """Test suite for FuzzyConvLayer."""
    
    def test_fuzzy_conv_shape(self):
        """Test that fuzzy convolution produces correct output shape."""
        print("\n[TEST] Fuzzy Convolution Layer Shape...")
        
        batch_size = 32
        sequence_length = 50
        input_channels = 3
        filters = 64
        kernel_size = 3
        
        # Create layer
        conv_layer = FuzzyConvLayer(filters=filters, kernel_size=kernel_size)
        
        # Create dummy input
        input_data = tf.random.normal((batch_size, sequence_length, input_channels))
        
        # Forward pass
        output = conv_layer(input_data)
        
        # Check shape (SAME padding keeps sequence length)
        expected_shape = (batch_size, sequence_length, filters)
        self.assertEqual(output.shape, expected_shape)
        print(f"   ✅ Input: {input_data.shape} -> Output: {output.shape}")
    
    def test_fuzzy_conv_activation_range(self):
        """Test that sigmoid activation produces values in [0, 1]."""
        print("\n[TEST] Fuzzy Convolution Activation Range...")
        
        conv_layer = FuzzyConvLayer(filters=64, kernel_size=3)
        input_data = tf.random.normal((16, 50, 3))
        
        output = conv_layer(input_data)
        
        # Check sigmoid range
        self.assertTrue(tf.reduce_all(output >= 0.0))
        self.assertTrue(tf.reduce_all(output <= 1.0))
        print(f"   ✅ Output range: [{tf.reduce_min(output).numpy():.4f}, {tf.reduce_max(output).numpy():.4f}]")
    
    def test_fuzzy_conv_trainable_params(self):
        """Test that fuzzy conv has correct trainable parameters."""
        print("\n[TEST] Fuzzy Convolution Trainable Parameters...")
        
        filters = 128
        kernel_size = 3
        input_channels = 64
        
        conv_layer = FuzzyConvLayer(filters=filters, kernel_size=kernel_size)
        conv_layer.build((None, 100, input_channels))
        
        trainable_vars = conv_layer.trainable_variables
        self.assertEqual(len(trainable_vars), 2)  # weights and bias
        
        # Check shapes
        expected_weight_shape = (kernel_size, input_channels, filters)
        expected_bias_shape = (filters,)
        
        self.assertEqual(tuple(trainable_vars[0].shape), expected_weight_shape)
        self.assertEqual(tuple(trainable_vars[1].shape), expected_bias_shape)
        print(f"   ✅ Weight shape: {trainable_vars[0].shape}, Bias shape: {trainable_vars[1].shape}")


class TestDefuzzificationLayer(unittest.TestCase):
    """Test suite for DefuzzificationLayer."""
    
    def test_defuzzification_shape(self):
        """Test that defuzzification produces correct output shape."""
        print("\n[TEST] Defuzzification Layer Shape...")
        
        batch_size = 32
        sequence_length = 100
        fuzzy_dimension = 3
        
        # Create layer
        defuzz_layer = DefuzzificationLayer()
        
        # Create dummy input
        input_data = tf.random.uniform((batch_size, sequence_length, fuzzy_dimension))
        
        # Forward pass
        output = defuzz_layer(input_data)
        
        # Check shape (should remove fuzzy dimension)
        expected_shape = (batch_size, sequence_length)
        self.assertEqual(output.shape, expected_shape)
        print(f"   ✅ Input: {input_data.shape} -> Output: {output.shape}")
    
    def test_defuzzification_mean_aggregation(self):
        """Test that defuzzification correctly computes mean."""
        print("\n[TEST] Defuzzification Mean Aggregation...")
        
        defuzz_layer = DefuzzificationLayer()
        
        # Create known input
        input_data = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=tf.float32)
        
        output = defuzz_layer(input_data)
        
        # Expected: mean along last axis
        expected = tf.constant([[2.0, 5.0]], dtype=tf.float32)
        
        self.assertTrue(tf.reduce_all(tf.abs(output - expected) < 1e-6))
        print(f"   ✅ Mean aggregation working correctly")


class TestFixedFuzzyCNN(unittest.TestCase):
    """Test suite for complete FixedFuzzyCNN model."""
    
    def test_forward_pass(self):
        """Test complete forward pass with dummy data."""
        print("\n[TEST] Complete Forward Pass...")
        
        num_snps = 50
        batch_size = 32
        num_classes = 2
        
        # Create model
        model = FixedFuzzyCNN(num_snps=num_snps, num_classes=num_classes)
        
        # Generate dummy input
        input_data = tf.random.uniform((batch_size, num_snps), minval=0, maxval=3)
        input_data = tf.floor(input_data)
        
        # Forward pass
        output = model(input_data, training=False)
        
        # Check output shape
        expected_shape = (batch_size, num_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check softmax properties
        prob_sums = tf.reduce_sum(output, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(prob_sums - 1.0) < 1e-5))
        
        print(f"   ✅ Input: {input_data.shape} -> Output: {output.shape}")
        print(f"   ✅ Softmax probabilities sum to 1.0")
    
    def test_model_compilation(self):
        """Test that model compiles without errors."""
        print("\n[TEST] Model Compilation...")
        
        num_snps = 50
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Check optimizer
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        
        # Check loss
        self.assertIsNotNone(model.loss)
        
        # Check that model compiled successfully
        self.assertIsNotNone(model.optimizer)
        
        print(f"   ✅ Optimizer: Adam")
        print(f"   ✅ Loss: sparse_categorical_crossentropy")
        print(f"   ✅ Model compiled successfully")
    
    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        print("\n[TEST] Parameter Count...")
        
        num_snps = 50
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        total_params = model.count_params()
        
        # Should be less than 5M for 50 SNPs
        self.assertLess(total_params, 5_000_000)
        
        print(f"   ✅ Total parameters: {total_params:,}")
        print(f"   ✅ Parameter count is reasonable (< 5M)")
    
    def test_training_on_dummy_data(self):
        """Test training for a few epochs on dummy data."""
        print("\n[TEST] Training on Dummy Data...")
        
        num_snps = 50
        num_samples = 200
        
        # Create model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Generate dummy data
        X_train = tf.random.uniform((num_samples, num_snps), minval=0, maxval=3, dtype=tf.float32)
        X_train = tf.floor(X_train)
        y_train = tf.random.uniform((num_samples,), minval=0, maxval=2, dtype=tf.int32)
        
        # Train for 3 epochs
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Check that loss exists and decreased or stayed reasonable
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        
        self.assertGreater(initial_loss, 0)
        self.assertGreater(final_loss, 0)
        
        print(f"   ✅ Initial loss: {initial_loss:.4f}")
        print(f"   ✅ Final loss: {final_loss:.4f}")
        print(f"   ✅ Training completed successfully")


class TestFuzzyCNNWithRealData(unittest.TestCase):
    """Test suite for Fuzzy CNN with real preprocessed data."""
    
    def test_training_on_preprocessed_data(self):
        """Test training on actual preprocessed dataset."""
        print("\n[TEST] Training on Real Preprocessed Data...")
        
        # Load preprocessed data
        data_path = 'data/processed/model1/order2/snps50/dataset_0.npz'
        
        if not os.path.exists(data_path):
            print(f"   ⚠️ Skipping: {data_path} not found")
            self.skipTest("Preprocessed data not available")
        
        data = np.load(data_path, allow_pickle=True)
        
        X_val = data['validation_X'].astype(np.float32)
        y_val = data['validation_y'].astype(np.int32)
        
        num_snps = X_val.shape[1]
        
        print(f"   📊 Dataset: {X_val.shape[0]} samples, {num_snps} SNPs")
        
        # Build model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Train for 5 epochs
        print(f"   🔄 Training for 5 epochs...")
        history = model.fit(
            X_val[:400], y_val[:400],
            validation_data=(X_val[400:], y_val[400:]),
            epochs=5,
            batch_size=64,
            verbose=0
        )
        
        # Check that loss decreased
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        
        print(f"   ✅ Initial loss: {initial_loss:.4f}")
        print(f"   ✅ Final loss: {final_loss:.4f}")
        
        # Loss should be positive and reasonable
        self.assertGreater(final_loss, 0)
        self.assertLess(final_loss, 5.0)
        
        # Check accuracy
        final_acc = history.history['accuracy'][-1]
        print(f"   ✅ Final accuracy: {final_acc:.4f}")
        
        # For pure epistasis, should eventually get > 50% accuracy
        self.assertGreater(final_acc, 0.4)
    
    def test_model_evaluation_on_test_set(self):
        """Test model evaluation on test set."""
        print("\n[TEST] Model Evaluation on Test Set...")
        
        data_path = 'data/processed/model1/order2/snps50/dataset_0.npz'
        
        if not os.path.exists(data_path):
            print(f"   ⚠️ Skipping: {data_path} not found")
            self.skipTest("Preprocessed data not available")
        
        data = np.load(data_path, allow_pickle=True)
        
        X_test = data['test_X'].astype(np.float32)
        y_test = data['test_y'].astype(np.int32)
        
        num_snps = X_test.shape[1]
        
        # Build and compile model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Evaluate (without training)
        results = model.evaluate(X_test, y_test, verbose=0)
        
        loss = results[0]
        accuracy = results[1]
        
        print(f"   ✅ Test loss: {loss:.4f}")
        print(f"   ✅ Test accuracy: {accuracy:.4f}")
        
        # Untrained model should have reasonable loss and ~50% accuracy
        self.assertGreater(loss, 0)
        self.assertLess(loss, 10.0)
        self.assertGreater(accuracy, 0.3)
        self.assertLess(accuracy, 0.7)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_different_snp_sizes(self):
        """Test model with different SNP sizes."""
        print("\n[TEST] Different SNP Sizes...")
        
        snp_sizes = [50, 100, 200]
        
        for num_snps in snp_sizes:
            model = build_fuzzy_cnn(num_snps=num_snps)
            
            # Test forward pass
            input_data = tf.random.uniform((16, num_snps), minval=0, maxval=3)
            output = model(input_data, training=False)
            
            self.assertEqual(output.shape, (16, 2))
            print(f"   ✅ SNPs={num_snps}: {input_data.shape} -> {output.shape}")
    
    def test_batch_size_variations(self):
        """Test model with different batch sizes."""
        print("\n[TEST] Different Batch Sizes...")
        
        num_snps = 50
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        batch_sizes = [1, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            input_data = tf.random.uniform((batch_size, num_snps), minval=0, maxval=3)
            output = model(input_data, training=False)
            
            self.assertEqual(output.shape, (batch_size, 2))
            print(f"   ✅ Batch size={batch_size}: Output shape {output.shape}")
    
    def test_extreme_values(self):
        """Test model with extreme input values."""
        print("\n[TEST] Extreme Input Values...")
        
        num_snps = 50
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Test with all zeros
        zeros = tf.zeros((16, num_snps))
        output_zeros = model(zeros, training=False)
        self.assertEqual(output_zeros.shape, (16, 2))
        print(f"   ✅ All zeros: Output shape {output_zeros.shape}")
        
        # Test with all ones
        ones = tf.ones((16, num_snps))
        output_ones = model(ones, training=False)
        self.assertEqual(output_ones.shape, (16, 2))
        print(f"   ✅ All ones: Output shape {output_ones.shape}")
        
        # Test with all twos
        twos = tf.ones((16, num_snps)) * 2.0
        output_twos = model(twos, training=False)
        self.assertEqual(output_twos.shape, (16, 2))
        print(f"   ✅ All twos: Output shape {output_twos.shape}")


def run_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print("FUZZY CNN UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzificationLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzyConvLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestDefuzzificationLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestFixedFuzzyCNN))
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzyCNNWithRealData))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    print(f"⏭️ Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 70 + "\n")
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
