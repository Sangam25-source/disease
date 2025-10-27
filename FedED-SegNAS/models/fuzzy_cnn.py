"""
Fuzzy CNN Architecture for Epistasis Detection
===============================================

This module implements a Fuzzy Convolutional Neural Network designed specifically
for detecting epistatic interactions in SNP data.

Key Components:
- FuzzificationLayer: Converts discrete SNP values to fuzzy membership values
- FuzzyConvLayer: Convolutional layer with fuzzy weights
- DefuzzificationLayer: Converts fuzzy outputs back to crisp values
- FixedFuzzyCNN: Complete model architecture

Author: FedED-SegNAS Project
Date: October 2024
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class FuzzificationLayer(layers.Layer):
    """
    Converts discrete SNP data to fuzzy membership values using Gaussian membership functions.
    
    For each SNP value (0, 1, 2), this layer computes membership degrees across
    multiple fuzzy sets (default 3 sets corresponding to genotypes AA, Aa, aa).
    
    Parameters:
    -----------
    num_fuzzy_sets : int, default=3
        Number of fuzzy sets per SNP (typically 3 for genotypes)
    
    Input Shape:
    -----------
    (batch_size, num_snps)
    
    Output Shape:
    ------------
    (batch_size, num_snps, num_fuzzy_sets)
    
    Example:
    --------
    >>> fuzz_layer = FuzzificationLayer(num_fuzzy_sets=3)
    >>> input_snps = tf.constant([[0.0, 1.0, 2.0]], shape=(1, 3))
    >>> output = fuzz_layer(input_snps)
    >>> print(output.shape)  # (1, 3, 3)
    """
    
    def __init__(self, num_fuzzy_sets=3, **kwargs):
        super(FuzzificationLayer, self).__init__(**kwargs)
        self.num_fuzzy_sets = num_fuzzy_sets
    
    def build(self, input_shape):
        """Initialize trainable parameters for Gaussian membership functions."""
        # Trainable means for each fuzzy set
        self.means = self.add_weight(
            name='means',
            shape=(self.num_fuzzy_sets,),
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=2.0),
            trainable=True
        )
        
        # Trainable standard deviations for each fuzzy set
        self.stds = self.add_weight(
            name='stds',
            shape=(self.num_fuzzy_sets,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        
        super(FuzzificationLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Compute fuzzy membership values using Gaussian membership functions.
        
        Formula: membership = exp(-((input - mean)^2) / (2 * std^2))
        """
        # Expand dimensions: (batch, snps) -> (batch, snps, 1)
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        
        # Reshape parameters for broadcasting
        means_expanded = tf.reshape(self.means, (1, 1, -1))
        stds_expanded = tf.reshape(self.stds, (1, 1, -1))
        
        # Prevent division by zero
        stds_expanded = tf.maximum(stds_expanded, 1e-8)
        
        # Gaussian membership function
        membership = tf.exp(
            -tf.square(inputs_expanded - means_expanded) / 
            (2 * tf.square(stds_expanded))
        )
        
        return membership
    
    def get_config(self):
        config = super(FuzzificationLayer, self).get_config()
        config.update({'num_fuzzy_sets': self.num_fuzzy_sets})
        return config


class FuzzyConvLayer(layers.Layer):
    """
    1D Convolutional layer designed for fuzzy data.
    
    Performs convolution on fuzzified SNP data to detect epistatic interactions.
    Uses sigmoid activation to maintain fuzzy logic properties.
    
    Parameters:
    -----------
    filters : int
        Number of output filters
    kernel_size : int
        Size of the convolutional kernel
    
    Input Shape:
    -----------
    (batch_size, sequence_length, num_fuzzy_sets)
    
    Output Shape:
    ------------
    (batch_size, sequence_length, filters)
    
    Example:
    --------
    >>> fuzzy_conv = FuzzyConvLayer(filters=64, kernel_size=3)
    >>> input_data = tf.random.normal((32, 50, 3))
    >>> output = fuzzy_conv(input_data)
    >>> print(output.shape)  # (32, 50, 64)
    """
    
    def __init__(self, filters, kernel_size, **kwargs):
        super(FuzzyConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        """Initialize convolutional weights and biases."""
        # Convolutional kernel
        self.conv_weights = self.add_weight(
            name='conv_weights',
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Bias term
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        super(FuzzyConvLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Perform fuzzy convolution with sigmoid activation.
        
        Sigmoid is used to maintain values in [0, 1] range suitable for fuzzy logic.
        """
        # 1D convolution
        output = tf.nn.conv1d(
            inputs, 
            filters=self.conv_weights, 
            stride=1, 
            padding='SAME'
        )
        
        # Add bias
        output = tf.nn.bias_add(output, self.bias)
        
        # Sigmoid activation for fuzzy compatibility
        return tf.nn.sigmoid(output)
    
    def get_config(self):
        config = super(FuzzyConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config


class DefuzzificationLayer(layers.Layer):
    """
    Converts fuzzy outputs back to crisp values using mean aggregation.
    
    This layer aggregates multiple fuzzy membership values into single crisp values
    by computing the mean across the fuzzy dimension.
    
    Input Shape:
    -----------
    (batch_size, sequence_length, fuzzy_dimension)
    
    Output Shape:
    ------------
    (batch_size, sequence_length)
    
    Example:
    --------
    >>> defuzz_layer = DefuzzificationLayer()
    >>> fuzzy_input = tf.random.uniform((32, 100, 3))
    >>> crisp_output = defuzz_layer(fuzzy_input)
    >>> print(crisp_output.shape)  # (32, 100)
    """
    
    def __init__(self, **kwargs):
        super(DefuzzificationLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        """Aggregate fuzzy values by computing mean across fuzzy dimension."""
        return tf.reduce_mean(inputs, axis=-1)
    
    def get_config(self):
        return super(DefuzzificationLayer, self).get_config()


class FixedFuzzyCNN(Model):
    """
    Complete Fuzzy CNN Architecture for Epistasis Detection.
    
    Architecture Overview:
    ----------------------
    1. Fuzzification Layer (SNP -> Fuzzy Sets)
    2. Three Fuzzy Convolutional Blocks:
       - Block 1: 64 filters, kernel=3, MaxPool(2), Dropout(0.3)
       - Block 2: 128 filters, kernel=3, MaxPool(2), Dropout(0.3)
       - Block 3: 256 filters, kernel=3, MaxPool(2)
    3. Defuzzification Layer (Fuzzy -> Crisp)
    4. Fully Connected Layers:
       - Dense(512, relu), Dropout(0.5)
       - Dense(256, relu), Dropout(0.5)
       - Dense(num_classes, softmax)
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_classes : int, default=2
        Number of output classes (typically 2 for case/control)
    
    Input Shape:
    -----------
    (batch_size, num_snps)
    
    Output Shape:
    ------------
    (batch_size, num_classes)
    
    Example:
    --------
    >>> model = FixedFuzzyCNN(num_snps=50, num_classes=2)
    >>> input_data = tf.random.uniform((32, 50), minval=0, maxval=3)
    >>> predictions = model(input_data, training=True)
    >>> print(predictions.shape)  # (32, 2)
    """
    
    def __init__(self, num_snps, num_classes=2, **kwargs):
        super(FixedFuzzyCNN, self).__init__(**kwargs)
        
        self.num_snps = num_snps
        self.num_classes = num_classes
        
        # Fuzzification
        self.fuzzification = FuzzificationLayer(num_fuzzy_sets=3)
        
        # Convolutional Block 1
        self.conv1 = FuzzyConvLayer(filters=64, kernel_size=3, name='fuzzy_conv1')
        self.pool1 = layers.MaxPooling1D(pool_size=2, name='pool1')
        self.dropout1 = layers.Dropout(0.3, name='dropout1')
        
        # Convolutional Block 2
        self.conv2 = FuzzyConvLayer(filters=128, kernel_size=3, name='fuzzy_conv2')
        self.pool2 = layers.MaxPooling1D(pool_size=2, name='pool2')
        self.dropout2 = layers.Dropout(0.3, name='dropout2')
        
        # Convolutional Block 3
        self.conv3 = FuzzyConvLayer(filters=256, kernel_size=3, name='fuzzy_conv3')
        self.pool3 = layers.MaxPooling1D(pool_size=2, name='pool3')
        
        # Defuzzification
        self.defuzzification = DefuzzificationLayer(name='defuzzification')
        
        # Fully Connected Layers
        self.flatten = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(512, activation='relu', name='dense1')
        self.dropout3 = layers.Dropout(0.5, name='dropout3')
        self.dense2 = layers.Dense(256, activation='relu', name='dense2')
        self.dropout4 = layers.Dropout(0.5, name='dropout4')
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='output')
    
    def call(self, inputs, training=False):
        """
        Forward pass through the Fuzzy CNN.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input SNP data with shape (batch_size, num_snps)
        training : bool
            Whether the model is in training mode (affects dropout)
        
        Returns:
        --------
        tf.Tensor
            Class probabilities with shape (batch_size, num_classes)
        """
        # Fuzzification: (batch, snps) -> (batch, snps, 3)
        x = self.fuzzification(inputs)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Defuzzification: (batch, seq, filters) -> (batch, seq)
        x = self.defuzzification(x)
        
        # Flatten and Dense Layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)
        x = self.dropout4(x, training=training)
        
        # Output
        return self.output_layer(x)
    
    def get_config(self):
        return {
            'num_snps': self.num_snps,
            'num_classes': self.num_classes
        }


def build_fuzzy_cnn(num_snps, num_classes=2, learning_rate=0.001):
    """
    Factory function to build and compile a Fuzzy CNN model.
    
    This function creates a FixedFuzzyCNN model, performs a dummy forward pass
    to initialize all layers, and compiles the model with appropriate loss and metrics.
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_classes : int, default=2
        Number of output classes
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    
    Returns:
    --------
    FixedFuzzyCNN
        Compiled Keras model ready for training
    
    Example:
    --------
    >>> model = build_fuzzy_cnn(num_snps=50)
    >>> model.summary()
    >>> history = model.fit(X_train, y_train, epochs=10)
    
    Notes:
    ------
    - Loss: sparse_categorical_crossentropy (for integer labels)
    - Optimizer: Adam
    - Metrics: accuracy, precision, recall
    """
    # Create model
    model = FixedFuzzyCNN(num_snps=num_snps, num_classes=num_classes)
    
    # Dummy forward pass to build all layers
    dummy_input = tf.random.normal((1, num_snps))
    _ = model(dummy_input, training=False)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("FUZZY CNN ARCHITECTURE TEST")
    print("=" * 70)
    
    # Test with 50 SNPs
    num_snps = 50
    batch_size = 32
    
    print(f"\nBuilding Fuzzy CNN with {num_snps} SNPs...")
    model = build_fuzzy_cnn(num_snps=num_snps)
    
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    model.summary()
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("TESTING FORWARD PASS")
    print("=" * 70)
    
    # Generate dummy SNP data (values 0, 1, 2)
    dummy_input = tf.random.uniform((batch_size, num_snps), minval=0, maxval=3, dtype=tf.float32)
    dummy_input = tf.floor(dummy_input)  # Ensure integer values
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input range: [{tf.reduce_min(dummy_input).numpy()}, {tf.reduce_max(dummy_input).numpy()}]")
    
    # Forward pass
    output = model(dummy_input, training=False)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output).numpy():.4f}, {tf.reduce_max(output).numpy():.4f}]")
    print(f"Sum of probabilities (should be ~1.0): {tf.reduce_sum(output[0]).numpy():.4f}")
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n✅ Total parameters: {total_params:,}")
    
    if total_params < 5_000_000:
        print("✅ Parameter count is reasonable (< 5M)")
    else:
        print("⚠️ Warning: Parameter count exceeds 5M")
    
    print("\n" + "=" * 70)
    print("✅ FUZZY CNN TEST COMPLETE")
    print("=" * 70)
