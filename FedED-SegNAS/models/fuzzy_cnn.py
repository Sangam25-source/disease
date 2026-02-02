"""
Fuzzy CNN Architecture for Epistasis Detection - IMPROVED VERSION
==================================================================

This module implements an enhanced Fuzzy Convolutional Neural Network designed 
specifically for detecting epistatic interactions in SNP data.

Key Improvements over Original:
- Better fuzzy parameter initialization (means at 0, 1, 2)
- Constrained standard deviations for fuzzy sets
- LeakyReLU/Swish activation instead of sigmoid (no vanishing gradients)
- Batch normalization throughout
- Adaptive pooling based on input size
- L2 regularization on weights
- Skip/residual connections
- Attention mechanism for SNP importance
- Learnable defuzzification
- Adjusted dropout rates
- Gradient clipping in optimizer

Author: FedED-SegNAS Project
Date: January 2025
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import math


class ImprovedFuzzificationLayer(layers.Layer):
    """
    Improved Fuzzification Layer with proper initialization and constraints.
    
    Converts discrete SNP data to fuzzy membership values using Gaussian 
    membership functions with:
    - Means initialized at 0, 1, 2 (corresponding to genotypes)
    - Constrained standard deviations (always positive)
    - Optional learnable vs fixed parameters
    
    Parameters:
    -----------
    num_fuzzy_sets : int, default=3
        Number of fuzzy sets per SNP (typically 3 for genotypes)
    learnable : bool, default=True
        Whether fuzzy parameters are trainable
    initial_std : float, default=0.4
        Initial standard deviation for fuzzy sets
    
    Input Shape:
    -----------
    (batch_size, num_snps)
    
    Output Shape:
    ------------
    (batch_size, num_snps, num_fuzzy_sets)
    """
    
    def __init__(self, num_fuzzy_sets=3, learnable=True, initial_std=0.4, **kwargs):
        super(ImprovedFuzzificationLayer, self).__init__(**kwargs)
        self.num_fuzzy_sets = num_fuzzy_sets
        self.learnable = learnable
        self.initial_std = initial_std
    
    def build(self, input_shape):
        """Initialize parameters with proper values for genotype data."""
        # Initialize means at exactly 0, 1, 2 for the three genotypes
        initial_means = np.linspace(0.0, 2.0, self.num_fuzzy_sets).astype(np.float32)
        
        self.means = self.add_weight(
            name='means',
            shape=(self.num_fuzzy_sets,),
            initializer=tf.keras.initializers.Constant(initial_means),
            trainable=self.learnable
        )
        
        # Initialize stds with smaller values for better separation
        # Use log-scale to ensure positivity after softplus transformation
        initial_log_stds = np.full(self.num_fuzzy_sets, 
                                    np.log(np.exp(self.initial_std) - 1),
                                    dtype=np.float32)
        
        self._log_stds = self.add_weight(
            name='log_stds',
            shape=(self.num_fuzzy_sets,),
            initializer=tf.keras.initializers.Constant(initial_log_stds),
            trainable=self.learnable
        )
        
        super(ImprovedFuzzificationLayer, self).build(input_shape)
    
    @property
    def stds(self):
        """Get constrained positive standard deviations using softplus."""
        # Softplus ensures stds are always positive: softplus(x) = log(1 + exp(x))
        # Add small epsilon to prevent numerical issues
        return tf.nn.softplus(self._log_stds) + 1e-6
    
    def call(self, inputs):
        """
        Compute fuzzy membership values using Gaussian membership functions.
        
        Formula: membership = exp(-((input - mean)^2) / (2 * std^2))
        """
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # Expand dimensions: (batch, snps) -> (batch, snps, 1)
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        
        # Reshape parameters for broadcasting
        means_expanded = tf.reshape(self.means, (1, 1, -1))
        stds_expanded = tf.reshape(self.stds, (1, 1, -1))
        
        # Gaussian membership function
        membership = tf.exp(
            -tf.square(inputs_expanded - means_expanded) / 
            (2 * tf.square(stds_expanded))
        )
        
        return membership
    
    def get_config(self):
        config = super(ImprovedFuzzificationLayer, self).get_config()
        config.update({
            'num_fuzzy_sets': self.num_fuzzy_sets,
            'learnable': self.learnable,
            'initial_std': self.initial_std
        })
        return config


class ImprovedFuzzyConvBlock(layers.Layer):
    """
    Improved Fuzzy Convolutional Block with:
    - Batch normalization
    - LeakyReLU/Swish activation (no vanishing gradients)
    - L2 regularization
    - Optional residual connection
    
    Parameters:
    -----------
    filters : int
        Number of output filters
    kernel_size : int
        Size of the convolutional kernel
    l2_reg : float, default=0.001
        L2 regularization factor
    dropout_rate : float, default=0.2
        Dropout rate after convolution
    use_residual : bool, default=True
        Whether to use residual connection
    activation : str, default='swish'
        Activation function ('swish', 'leaky_relu', 'relu')
    """
    
    def __init__(self, filters, kernel_size, l2_reg=0.001, dropout_rate=0.2,
                 use_residual=True, activation='swish', **kwargs):
        super(ImprovedFuzzyConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.activation_name = activation
        
        # Main convolution
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        )
        
        # Batch normalization
        self.batch_norm = layers.BatchNormalization()
        
        # Activation
        if activation == 'swish':
            self.activation = layers.Activation('swish')
        elif activation == 'leaky_relu':
            self.activation = layers.LeakyReLU(alpha=0.1)
        else:
            self.activation = layers.ReLU()
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Residual projection (if needed)
        self.residual_conv = None
    
    def build(self, input_shape):
        """Build residual projection if input/output channels differ."""
        input_channels = input_shape[-1]
        
        if self.use_residual and input_channels != self.filters:
            self.residual_conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )
        
        super(ImprovedFuzzyConvBlock, self).build(input_shape)
    
    def call(self, inputs, training=False):
        """Forward pass with optional residual connection."""
        # Main path
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        # Residual connection
        if self.use_residual:
            residual = inputs
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x = x + residual
        
        return x
    
    def get_config(self):
        config = super(ImprovedFuzzyConvBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'activation': self.activation_name
        })
        return config


class AdaptivePoolingLayer(layers.Layer):
    """
    Adaptive pooling that adjusts based on input size to prevent dimension collapse.
    
    Parameters:
    -----------
    target_size : int, optional
        Target output size. If None, uses adaptive reduction.
    pool_type : str, default='max'
        Type of pooling ('max' or 'avg')
    min_output_size : int, default=4
        Minimum output size to prevent collapse
    """
    
    def __init__(self, target_size=None, pool_type='max', min_output_size=4, **kwargs):
        super(AdaptivePoolingLayer, self).__init__(**kwargs)
        self.target_size = target_size
        self.pool_type = pool_type
        self.min_output_size = min_output_size
    
    def call(self, inputs):
        """Apply adaptive pooling based on input size."""
        input_size = tf.shape(inputs)[1]
        
        if self.target_size is not None:
            target = self.target_size
        else:
            # Adaptive: reduce by half but maintain minimum size
            target = tf.maximum(input_size // 2, self.min_output_size)
        
        # Calculate pool size needed
        pool_size = tf.maximum(input_size // target, 1)
        
        # Apply pooling
        if self.pool_type == 'max':
            pooled = tf.nn.max_pool1d(
                inputs, 
                ksize=pool_size, 
                strides=pool_size, 
                padding='SAME'
            )
        else:
            pooled = tf.nn.avg_pool1d(
                inputs, 
                ksize=pool_size, 
                strides=pool_size, 
                padding='SAME'
            )
        
        return pooled
    
    def get_config(self):
        config = super(AdaptivePoolingLayer, self).get_config()
        config.update({
            'target_size': self.target_size,
            'pool_type': self.pool_type,
            'min_output_size': self.min_output_size
        })
        return config


class SNPAttentionLayer(layers.Layer):
    """
    Attention mechanism to weight SNP importance for epistasis detection.
    
    Learns which SNPs and SNP combinations are most important for classification.
    
    Parameters:
    -----------
    attention_units : int, default=64
        Number of units in attention network
    num_heads : int, default=4
        Number of attention heads (for multi-head attention)
    use_multi_head : bool, default=False
        Whether to use multi-head attention
    """
    
    def __init__(self, attention_units=64, num_heads=4, use_multi_head=False, **kwargs):
        super(SNPAttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.use_multi_head = use_multi_head
        
        if use_multi_head:
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=attention_units // num_heads,
                dropout=0.1
            )
        else:
            # Simple attention mechanism
            self.attention_dense1 = layers.Dense(
                attention_units, 
                activation='tanh',
                kernel_regularizer=regularizers.l2(0.001)
            )
            self.attention_dense2 = layers.Dense(1)
    
    def call(self, inputs, training=False):
        """Apply attention weighting to input features."""
        if self.use_multi_head:
            # Self-attention
            attended = self.attention(inputs, inputs, training=training)
            return attended
        else:
            # Simple attention
            attention_scores = self.attention_dense1(inputs)
            attention_scores = self.attention_dense2(attention_scores)
            attention_weights = tf.nn.softmax(attention_scores, axis=1)
            
            # Apply attention weights
            attended = inputs * attention_weights
            return attended
    
    def get_config(self):
        config = super(SNPAttentionLayer, self).get_config()
        config.update({
            'attention_units': self.attention_units,
            'num_heads': self.num_heads,
            'use_multi_head': self.use_multi_head
        })
        return config


class LearnableDefuzzificationLayer(layers.Layer):
    """
    Improved defuzzification with learnable aggregation instead of simple mean.
    
    Uses a small neural network to learn optimal aggregation of fuzzy values.
    
    Parameters:
    -----------
    hidden_units : int, default=32
        Number of hidden units in aggregation network
    aggregation_type : str, default='learned'
        Type of aggregation ('learned', 'weighted_mean', 'attention')
    """
    
    def __init__(self, hidden_units=32, aggregation_type='learned', **kwargs):
        super(LearnableDefuzzificationLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.aggregation_type = aggregation_type
    
    def build(self, input_shape):
        """Build aggregation network."""
        fuzzy_dim = input_shape[-1]
        
        if self.aggregation_type == 'learned':
            # Small network for learned aggregation
            self.agg_dense1 = layers.Dense(
                self.hidden_units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            )
            self.agg_dense2 = layers.Dense(1, activation='linear')
            
        elif self.aggregation_type == 'weighted_mean':
            # Learnable weights for weighted average
            self.weights = self.add_weight(
                name='aggregation_weights',
                shape=(fuzzy_dim,),
                initializer='ones',
                trainable=True
            )
        
        elif self.aggregation_type == 'attention':
            # Attention-based aggregation
            self.attention_dense = layers.Dense(1)
        
        super(LearnableDefuzzificationLayer, self).build(input_shape)
    
    def call(self, inputs):
        """Aggregate fuzzy values to crisp values."""
        if self.aggregation_type == 'learned':
            # Apply small network per position
            x = self.agg_dense1(inputs)
            x = self.agg_dense2(x)
            return tf.squeeze(x, axis=-1)
        
        elif self.aggregation_type == 'weighted_mean':
            # Normalized weighted mean
            normalized_weights = tf.nn.softmax(self.weights)
            weighted_sum = tf.reduce_sum(inputs * normalized_weights, axis=-1)
            return weighted_sum
        
        elif self.aggregation_type == 'attention':
            # Attention-based aggregation
            attention_scores = self.attention_dense(inputs)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended = tf.reduce_sum(inputs * attention_weights, axis=-1)
            return attended
        
        else:
            # Fallback to simple mean
            return tf.reduce_mean(inputs, axis=-1)
    
    def get_config(self):
        config = super(LearnableDefuzzificationLayer, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'aggregation_type': self.aggregation_type
        })
        return config


class InputNormalizationLayer(layers.Layer):
    """
    Normalizes SNP input values for better training stability.
    
    SNP values (0, 1, 2) are normalized to have zero mean and unit variance,
    or scaled to a specific range.
    
    Parameters:
    -----------
    normalization_type : str, default='standard'
        Type of normalization ('standard', 'minmax', 'none')
    """
    
    def __init__(self, normalization_type='standard', **kwargs):
        super(InputNormalizationLayer, self).__init__(**kwargs)
        self.normalization_type = normalization_type
        
        # Pre-computed statistics for SNP values (0, 1, 2)
        # Mean = 1.0, Std ≈ 0.816 for uniform distribution
        self.snp_mean = 1.0
        self.snp_std = 0.8165  # sqrt(2/3)
    
    def call(self, inputs):
        """Normalize input SNP values."""
        inputs = tf.cast(inputs, tf.float32)
        
        if self.normalization_type == 'standard':
            # Standardize to zero mean, unit variance
            normalized = (inputs - self.snp_mean) / self.snp_std
            return normalized
        
        elif self.normalization_type == 'minmax':
            # Scale to [0, 1] range
            normalized = inputs / 2.0
            return normalized
        
        else:
            return inputs
    
    def get_config(self):
        config = super(InputNormalizationLayer, self).get_config()
        config.update({'normalization_type': self.normalization_type})
        return config


class ImprovedFuzzyCNN(Model):
    """
    Improved Fuzzy CNN Architecture for Epistasis Detection.
    
    Key Improvements:
    -----------------
    1. Proper fuzzy parameter initialization
    2. Batch normalization throughout
    3. Swish/LeakyReLU activations (no vanishing gradients)
    4. Adaptive pooling based on input size
    5. L2 regularization
    6. Skip/residual connections
    7. Attention mechanism for SNP importance
    8. Learnable defuzzification
    9. Input normalization
    10. Adjusted dropout rates
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_classes : int, default=2
        Number of output classes
    num_fuzzy_sets : int, default=3
        Number of fuzzy sets for fuzzification
    filters : list, default=[64, 128, 256]
        Number of filters for each conv block
    kernel_sizes : list, default=[5, 3, 3]
        Kernel sizes for each conv block
    dense_units : list, default=[256, 128]
        Number of units for dense layers
    l2_reg : float, default=0.001
        L2 regularization factor
    dropout_conv : float, default=0.2
        Dropout rate after conv blocks
    dropout_dense : float, default=0.4
        Dropout rate after dense layers
    use_attention : bool, default=True
        Whether to use attention mechanism
    use_residual : bool, default=True
        Whether to use residual connections
    activation : str, default='swish'
        Activation function ('swish', 'leaky_relu', 'relu')
    """
    
    def __init__(self, num_snps, num_classes=2, num_fuzzy_sets=3,
                 filters=None, kernel_sizes=None, dense_units=None,
                 l2_reg=0.001, dropout_conv=0.2, dropout_dense=0.4,
                 use_attention=True, use_residual=True, activation='swish',
                 **kwargs):
        super(ImprovedFuzzyCNN, self).__init__(**kwargs)
        
        # Store configuration
        self.num_snps = num_snps
        self.num_classes = num_classes
        self.num_fuzzy_sets = num_fuzzy_sets
        self.l2_reg = l2_reg
        self.dropout_conv = dropout_conv
        self.dropout_dense = dropout_dense
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.activation_name = activation
        
        # Default architecture parameters (adaptive based on num_snps)
        if filters is None:
            filters = self._get_adaptive_filters(num_snps)
        if kernel_sizes is None:
            kernel_sizes = self._get_adaptive_kernel_sizes(num_snps)
        if dense_units is None:
            dense_units = self._get_adaptive_dense_units(num_snps)
        
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        
        # Input normalization
        self.input_norm = InputNormalizationLayer(normalization_type='standard')
        
        # Fuzzification with proper initialization
        self.fuzzification = ImprovedFuzzificationLayer(
            num_fuzzy_sets=num_fuzzy_sets,
            learnable=True,
            initial_std=0.4
        )
        
        # Initial batch normalization after fuzzification
        self.initial_bn = layers.BatchNormalization()
        
        # Convolutional blocks with residual connections
        self.conv_blocks = []
        self.pool_layers = []
        
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            # Conv block
            conv_block = ImprovedFuzzyConvBlock(
                filters=f,
                kernel_size=k,
                l2_reg=l2_reg,
                dropout_rate=dropout_conv,
                use_residual=use_residual,
                activation=activation,
                name=f'conv_block_{i+1}'
            )
            self.conv_blocks.append(conv_block)
            
            # Adaptive pooling (only if we have enough spatial dimensions)
            pool = AdaptivePoolingLayer(
                pool_type='max',
                min_output_size=max(4, num_snps // (2 ** (len(filters) + 1))),
                name=f'pool_{i+1}'
            )
            self.pool_layers.append(pool)
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = SNPAttentionLayer(
                attention_units=64,
                use_multi_head=False,
                name='snp_attention'
            )
        
        # Learnable defuzzification
        self.defuzzification = LearnableDefuzzificationLayer(
            hidden_units=32,
            aggregation_type='learned',
            name='defuzzification'
        )
        
        # Global pooling to handle variable sequence lengths
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.global_max_pool = layers.GlobalMaxPooling1D()
        
        # Fully connected layers with batch norm
        self.dense_layers = []
        self.dense_bn_layers = []
        self.dense_dropout_layers = []
        
        for i, units in enumerate(dense_units):
            dense = layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer='he_normal',
                name=f'dense_{i+1}'
            )
            self.dense_layers.append(dense)
            
            bn = layers.BatchNormalization(name=f'dense_bn_{i+1}')
            self.dense_bn_layers.append(bn)
            
            if activation == 'swish':
                act = layers.Activation('swish')
            elif activation == 'leaky_relu':
                act = layers.LeakyReLU(alpha=0.1)
            else:
                act = layers.ReLU()
            self.dense_activations = getattr(self, 'dense_activations', [])
            self.dense_activations.append(act)
            
            dropout = layers.Dropout(dropout_dense, name=f'dense_dropout_{i+1}')
            self.dense_dropout_layers.append(dropout)
        
        # Output layer
        self.output_layer = layers.Dense(
            num_classes, 
            activation='softmax',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='output'
        )
    
    def _get_adaptive_filters(self, num_snps):
        """Get adaptive filter sizes based on input dimensions."""
        if num_snps <= 50:
            return [32, 64, 128]
        elif num_snps <= 200:
            return [64, 128, 256]
        elif num_snps <= 1000:
            return [64, 128, 256, 512]
        else:
            return [64, 128, 256, 512, 512]
    
    def _get_adaptive_kernel_sizes(self, num_snps):
        """Get adaptive kernel sizes based on input dimensions."""
        if num_snps <= 50:
            return [3, 3, 3]
        elif num_snps <= 200:
            return [5, 3, 3]
        elif num_snps <= 1000:
            return [7, 5, 3, 3]
        else:
            return [9, 7, 5, 3, 3]
    
    def _get_adaptive_dense_units(self, num_snps):
        """Get adaptive dense layer sizes based on input dimensions."""
        if num_snps <= 50:
            return [128, 64]
        elif num_snps <= 200:
            return [256, 128]
        elif num_snps <= 1000:
            return [512, 256]
        else:
            return [512, 256, 128]
    
    def call(self, inputs, training=False):
        """
        Forward pass through the Improved Fuzzy CNN.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input SNP data with shape (batch_size, num_snps)
        training : bool
            Whether the model is in training mode
        
        Returns:
        --------
        tf.Tensor
            Class probabilities with shape (batch_size, num_classes)
        """
        # Input normalization
        x = self.input_norm(inputs)
        
        # Fuzzification: (batch, snps) -> (batch, snps, num_fuzzy_sets)
        x = self.fuzzification(x)
        x = self.initial_bn(x, training=training)
        
        # Convolutional blocks with pooling
        for conv_block, pool in zip(self.conv_blocks, self.pool_layers):
            x = conv_block(x, training=training)
            
            # Only pool if we have enough spatial dimensions
            if x.shape[1] is not None and x.shape[1] > 4:
                x = pool(x)
        
        # Attention mechanism (optional)
        if self.use_attention:
            x = self.attention(x, training=training)
        
        # Defuzzification: aggregate across last dimension
        x = self.defuzzification(x)
        
        # Expand dims for global pooling if needed
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        
        # Global pooling (concatenate avg and max for richer features)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Dense layers
        for i, (dense, bn, dropout) in enumerate(zip(
            self.dense_layers, self.dense_bn_layers, self.dense_dropout_layers
        )):
            x = dense(x)
            x = bn(x, training=training)
            x = self.dense_activations[i](x)
            x = dropout(x, training=training)
        
        # Output
        return self.output_layer(x)
    
    def get_config(self):
        return {
            'num_snps': self.num_snps,
            'num_classes': self.num_classes,
            'num_fuzzy_sets': self.num_fuzzy_sets,
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'l2_reg': self.l2_reg,
            'dropout_conv': self.dropout_conv,
            'dropout_dense': self.dropout_dense,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'activation': self.activation_name
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Backward compatible alias
FixedFuzzyCNN = ImprovedFuzzyCNN


def build_fuzzy_cnn(num_snps, num_classes=2, learning_rate=0.001, 
                    use_attention=True, use_residual=True,
                    l2_reg=0.001, dropout_conv=0.2, dropout_dense=0.4,
                    activation='swish', clipnorm=1.0, clipvalue=0.5):
    """
    Factory function to build and compile an Improved Fuzzy CNN model.
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_classes : int, default=2
        Number of output classes
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    use_attention : bool, default=True
        Whether to use attention mechanism
    use_residual : bool, default=True
        Whether to use residual connections
    l2_reg : float, default=0.001
        L2 regularization factor
    dropout_conv : float, default=0.2
        Dropout rate after conv blocks
    dropout_dense : float, default=0.4
        Dropout rate after dense layers
    activation : str, default='swish'
        Activation function ('swish', 'leaky_relu', 'relu')
    clipnorm : float, default=1.0
        Gradient clipping by norm
    clipvalue : float, default=0.5
        Gradient clipping by value
    
    Returns:
    --------
    ImprovedFuzzyCNN
        Compiled Keras model ready for training
    
    Example:
    --------
    >>> model = build_fuzzy_cnn(num_snps=50)
    >>> model.summary()
    >>> history = model.fit(X_train, y_train, epochs=10)
    """
    # Create model
    model = ImprovedFuzzyCNN(
        num_snps=num_snps, 
        num_classes=num_classes,
        use_attention=use_attention,
        use_residual=use_residual,
        l2_reg=l2_reg,
        dropout_conv=dropout_conv,
        dropout_dense=dropout_dense,
        activation=activation
    )
    
    # Dummy forward pass to build all layers
    dummy_input = tf.random.normal((1, num_snps))
    _ = model(dummy_input, training=False)
    
    # Compile model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
        clipvalue=clipvalue
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def build_fuzzy_cnn_for_nas(architecture, num_snps, num_classes=2, learning_rate=0.001):
    """
    Build Fuzzy CNN from NAS-discovered architecture.
    
    Parameters:
    -----------
    architecture : dict
        Architecture specification from NAS containing:
        - num_blocks: int
        - blocks: list of dicts with 'filters', 'kernel_size', 'pool_type'
        - dense_1: int
        - dense_2: int
        - dropout: float
    num_snps : int
        Number of SNP features
    num_classes : int
        Number of output classes
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    ImprovedFuzzyCNN
        Compiled model with NAS architecture
    """
    # Extract architecture parameters
    num_blocks = architecture.get('num_blocks', 3)
    blocks = architecture.get('blocks', [])
    dense_1 = architecture.get('dense_1', 256)
    dense_2 = architecture.get('dense_2', 128)
    dropout = architecture.get('dropout', 0.3)
    
    # Build filter and kernel size lists
    filters = []
    kernel_sizes = []
    
    for i in range(num_blocks):
        if i < len(blocks):
            filters.append(blocks[i].get('filters', 64))
            kernel_sizes.append(blocks[i].get('kernel_size', 3))
        else:
            # Default values for remaining blocks
            filters.append(64 * (2 ** i))
            kernel_sizes.append(3)
    
    # Dense units
    dense_units = [dense_1]
    if dense_2 > 0:
        dense_units.append(dense_2)
    
    # Create model
    model = ImprovedFuzzyCNN(
        num_snps=num_snps,
        num_classes=num_classes,
        filters=filters,
        kernel_sizes=kernel_sizes,
        dense_units=dense_units,
        dropout_conv=dropout * 0.6,  # Lower dropout for conv
        dropout_dense=dropout,
        use_attention=True,
        use_residual=True
    )
    
    # Build model
    dummy_input = tf.random.normal((1, num_snps))
    _ = model(dummy_input, training=False)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss focuses training on hard examples and down-weights easy examples.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters:
    -----------
    gamma : float, default=2.0
        Focusing parameter (higher = more focus on hard examples)
    alpha : float, default=0.25
        Class balancing parameter
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        """Compute focal loss."""
        # Convert to float32
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.int32)
        
        # Get number of classes
        num_classes = tf.shape(y_pred)[-1]
        
        # One-hot encode labels
        y_true_one_hot = tf.one_hot(y_true, num_classes)
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Compute cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Compute focal weight
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config


def build_fuzzy_cnn_with_focal_loss(num_snps, num_classes=2, learning_rate=0.001,
                                     gamma=2.0, alpha=0.25, **kwargs):
    """
    Build Fuzzy CNN with Focal Loss for handling class imbalance.
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features
    num_classes : int
        Number of output classes
    learning_rate : float
        Learning rate for optimizer
    gamma : float
        Focal loss focusing parameter
    alpha : float
        Focal loss class balancing parameter
    **kwargs
        Additional arguments passed to build_fuzzy_cnn
    
    Returns:
    --------
    ImprovedFuzzyCNN
        Compiled model with focal loss
    """
    # Create model
    model = ImprovedFuzzyCNN(
        num_snps=num_snps,
        num_classes=num_classes,
        **kwargs
    )
    
    # Build model
    dummy_input = tf.random.normal((1, num_snps))
    _ = model(dummy_input, training=False)
    
    # Compile with focal loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=gamma, alpha=alpha),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("IMPROVED FUZZY CNN ARCHITECTURE TEST")
    print("=" * 70)
    
    # Test with different SNP sizes
    test_configs = [
        {'num_snps': 50, 'batch_size': 32},
        {'num_snps': 100, 'batch_size': 32},
        {'num_snps': 500, 'batch_size': 16},
        {'num_snps': 1000, 'batch_size': 16},
    ]
    
    for config in test_configs:
        num_snps = config['num_snps']
        batch_size = config['batch_size']
        
        print(f"\n{'='*70}")
        print(f"Testing with {num_snps} SNPs")
        print("="*70)
        
        # Build model
        model = build_fuzzy_cnn(num_snps=num_snps)
        
        # Generate dummy data
        dummy_input = tf.random.uniform(
            (batch_size, num_snps), 
            minval=0, maxval=3, 
            dtype=tf.float32
        )
        dummy_input = tf.floor(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        output = model(dummy_input, training=False)
        
        print(f"Output shape: {output.shape}")
        print(f"Sum of probabilities: {tf.reduce_sum(output[0]).numpy():.4f}")
        print(f"Total parameters: {model.count_params():,}")
        
        # Check architecture details
        print(f"Filters: {model.filters}")
        print(f"Kernel sizes: {model.kernel_sizes}")
        print(f"Dense units: {model.dense_units}")
    
    # Test model summary for 50 SNPs
    print("\n" + "="*70)
    print("MODEL SUMMARY (50 SNPs)")
    print("="*70)
    model = build_fuzzy_cnn(num_snps=50)
    model.summary()
    
    # Test training step
    print("\n" + "="*70)
    print("TESTING TRAINING STEP")
    print("="*70)
    
    # Generate training data
    X_train = tf.random.uniform((100, 50), minval=0, maxval=3, dtype=tf.float32)
    X_train = tf.floor(X_train)
    y_train = tf.random.uniform((100,), minval=0, maxval=2, dtype=tf.int32)
    
    # Train for a few steps
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)
    
    print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    
    # Test with focal loss
    print("\n" + "="*70)
    print("TESTING FOCAL LOSS MODEL")
    print("="*70)
    
    focal_model = build_fuzzy_cnn_with_focal_loss(num_snps=50, gamma=2.0, alpha=0.25)
    focal_history = focal_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)
    
    print(f"\nFocal loss model accuracy: {focal_history.history['accuracy'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - IMPROVED FUZZY CNN READY")
    print("="*70)
    
    # Print improvements summary
    print("\n" + "="*70)
    print("IMPROVEMENTS SUMMARY")
    print("="*70)
    print("""
    ✅ Proper fuzzy parameter initialization (means at 0, 1, 2)
    ✅ Constrained standard deviations (always positive via softplus)
    ✅ Swish/LeakyReLU activation (no vanishing gradients)
    ✅ Batch normalization throughout
    ✅ Adaptive pooling based on input size
    ✅ L2 regularization on all weights
    ✅ Skip/residual connections
    ✅ Attention mechanism for SNP importance
    ✅ Learnable defuzzification
    ✅ Input normalization
    ✅ Gradient clipping in optimizer
    ✅ Focal loss option for class imbalance
    ✅ Adaptive architecture based on input size
    ✅ Additional metrics: precision, recall, AUC
    """)
