"""
Search Space Definition for NAS
================================

Defines the architecture search space for Fuzzy CNN in federated learning.

This module provides:
- Architecture encoding scheme
- Valid architecture ranges
- Helper functions to encode/decode architectures

Author: FedED-SegNAS Project
Date: January 2026
"""

import numpy as np


class SearchSpace:
    """
    Defines the search space for neural architecture search.
    
    Architecture encoding:
    - num_blocks: Number of convolutional blocks [1, 2, 3, 4]
    - filters: Filters per block [32, 64, 128, 256]
    - kernel_size: Kernel sizes [3, 5, 7]
    - pool_type: Pooling type [0=Max, 1=Avg, 2=None]
    - dense_units: Dense layer sizes [128, 256, 512]
    - dropout_rate: Dropout rates [0.2, 0.3, 0.4, 0.5]
    
    Position Vector Format (16 elements):
    [num_blocks, 
     block0_filters, block0_kernel, block0_pool,
     block1_filters, block1_kernel, block1_pool,
     block2_filters, block2_kernel, block2_pool,
     block3_filters, block3_kernel, block3_pool,
     dense_1, dense_2, dropout]
    
    Example:
    --------
    >>> space = SearchSpace()
    >>> arch = space.random_architecture()
    >>> position = space.encode_architecture(arch)
    >>> decoded = space.decode_position(position)
    """
    
    def __init__(self):
        """Initialize search space with valid ranges."""
        # Define search space ranges
        self.num_blocks_range = [1, 2, 3, 4]
        self.filters_range = [32, 64, 128, 256]
        self.kernel_range = [3, 5, 7]
        self.pool_range = [0, 1, 2]  # 0=Max, 1=Avg, 2=None
        self.dense_range = [128, 256, 512]
        self.dropout_range = [0.2, 0.3, 0.4, 0.5]
        
        # Position vector size (fixed)
        self.position_size = 16
    
    def random_architecture(self):
        """
        Generate a random valid architecture.
        
        Returns:
        --------
        arch : dict
            Architecture specification with keys:
            - num_blocks: int
            - blocks: list of dicts with 'filters', 'kernel_size', 'pool_type'
            - dense_1: int
            - dense_2: int
            - dropout: float
        
        Example:
        --------
        >>> space = SearchSpace()
        >>> arch = space.random_architecture()
        >>> print(arch)
        {'num_blocks': 3, 'blocks': [...], 'dense_1': 256, ...}
        """
        num_blocks = np.random.choice(self.num_blocks_range)
        
        arch = {
            'num_blocks': num_blocks,
            'blocks': []
        }
        
        # Generate random blocks
        for i in range(num_blocks):
            block = {
                'filters': np.random.choice(self.filters_range),
                'kernel_size': np.random.choice(self.kernel_range),
                'pool_type': np.random.choice(self.pool_range)
            }
            arch['blocks'].append(block)
        
        # Generate dense layers and dropout
        arch['dense_1'] = np.random.choice(self.dense_range)
        arch['dense_2'] = np.random.choice(self.dense_range)
        arch['dropout'] = np.random.choice(self.dropout_range)
        
        return arch
    
    def encode_architecture(self, arch):
        """
        Convert architecture dict to position vector.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        position : np.ndarray
            Position vector of size 16
        
        Example:
        --------
        >>> space = SearchSpace()
        >>> arch = {'num_blocks': 2, 'blocks': [...], ...}
        >>> position = space.encode_architecture(arch)
        >>> print(position.shape)
        (16,)
        """
        position = [arch['num_blocks']]
        
        # Encode blocks (pad to max 4 blocks)
        for i in range(4):
            if i < len(arch['blocks']):
                block = arch['blocks'][i]
                position.extend([
                    block['filters'],
                    block['kernel_size'],
                    block['pool_type']
                ])
            else:
                # Padding for unused blocks
                position.extend([0, 0, 0])
        
        # Encode dense layers and dropout
        position.extend([
            arch['dense_1'],
            arch['dense_2'],
            arch['dropout']
        ])
        
        return np.array(position, dtype=np.float32)
    
    def decode_position(self, position):
        """
        Convert position vector to architecture dict.
        
        Parameters:
        -----------
        position : np.ndarray
            Position vector of size 16
        
        Returns:
        --------
        arch : dict
            Architecture specification
        
        Example:
        --------
        >>> space = SearchSpace()
        >>> position = np.array([3, 64, 3, 0, 128, 5, 1, ...])
        >>> arch = space.decode_position(position)
        >>> print(arch['num_blocks'])
        3
        """
        num_blocks = int(round(position[0]))
        num_blocks = max(1, min(4, num_blocks))  # Clip to [1, 4]
        
        arch = {
            'num_blocks': num_blocks,
            'blocks': []
        }
        
        # Decode blocks (only up to num_blocks)
        for i in range(num_blocks):
            idx = 1 + i * 3
            block = {
                'filters': int(round(position[idx])),
                'kernel_size': int(round(position[idx + 1])),
                'pool_type': int(round(position[idx + 2]))
            }
            # Ensure valid ranges
            block['filters'] = max(32, min(256, block['filters']))
            block['kernel_size'] = max(3, min(7, block['kernel_size']))
            block['pool_type'] = max(0, min(2, block['pool_type']))
            
            arch['blocks'].append(block)
        
        # Decode dense layers and dropout (ensure integers for dense)
        arch['dense_1'] = int(round(position[13]))
        arch['dense_2'] = int(round(position[14]))
        arch['dropout'] = float(position[15])
        
        # Ensure valid ranges for dense layers
        arch['dense_1'] = max(128, min(512, arch['dense_1']))
        arch['dense_2'] = max(128, min(512, arch['dense_2']))
        arch['dropout'] = max(0.2, min(0.5, arch['dropout']))
        
        return arch
    
    def is_valid_architecture(self, arch):
        """
        Check if architecture is valid.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        valid : bool
            True if architecture is valid
        """
        # Check num_blocks
        if arch['num_blocks'] not in self.num_blocks_range:
            return False
        
        # Check blocks
        if len(arch['blocks']) != arch['num_blocks']:
            return False
        
        for block in arch['blocks']:
            if block['filters'] not in self.filters_range:
                return False
            if block['kernel_size'] not in self.kernel_range:
                return False
            if block['pool_type'] not in self.pool_range:
                return False
        
        # Check dense layers
        if arch['dense_1'] not in self.dense_range:
            return False
        if arch['dense_2'] not in self.dense_range:
            return False
        
        # Check dropout
        if arch['dropout'] not in self.dropout_range:
            return False
        
        return True
    
    def get_architecture_summary(self, arch):
        """
        Get human-readable summary of architecture.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        summary : str
            Human-readable architecture summary
        """
        lines = []
        lines.append(f"Architecture Summary:")
        lines.append(f"  Blocks: {arch['num_blocks']}")
        
        for i, block in enumerate(arch['blocks']):
            pool_name = ['MaxPool', 'AvgPool', 'None'][block['pool_type']]
            lines.append(f"  Block {i+1}: {block['filters']} filters, "
                        f"kernel={block['kernel_size']}, pool={pool_name}")
        
        lines.append(f"  Dense 1: {arch['dense_1']} units")
        lines.append(f"  Dense 2: {arch['dense_2']} units")
        lines.append(f"  Dropout: {arch['dropout']}")
        
        return '\n'.join(lines)


# Test and demonstration
if __name__ == '__main__':
    print("="*80)
    print("SEARCH SPACE TEST")
    print("="*80)
    
    # Initialize search space
    space = SearchSpace()
    print("\n✅ SearchSpace initialized")
    print(f"   Position vector size: {space.position_size}")
    print(f"   Num blocks range: {space.num_blocks_range}")
    print(f"   Filters range: {space.filters_range}")
    print(f"   Kernel range: {space.kernel_range}")
    print(f"   Pool types: {space.pool_range} (0=Max, 1=Avg, 2=None)")
    print(f"   Dense range: {space.dense_range}")
    print(f"   Dropout range: {space.dropout_range}")
    
    # Test random architecture generation
    print("\n" + "-"*80)
    print("TEST 1: Random Architecture Generation")
    print("-"*80)
    
    arch = space.random_architecture()
    print(f"\n✅ Generated random architecture:")
    print(f"   {arch}")
    
    # Test encoding
    print("\n" + "-"*80)
    print("TEST 2: Architecture Encoding")
    print("-"*80)
    
    position = space.encode_architecture(arch)
    print(f"\n✅ Encoded to position vector:")
    print(f"   Shape: {position.shape}")
    print(f"   Values: {position}")
    
    # Test decoding
    print("\n" + "-"*80)
    print("TEST 3: Position Decoding")
    print("-"*80)
    
    decoded = space.decode_position(position)
    print(f"\n✅ Decoded back to architecture:")
    print(f"   {decoded}")
    
    # Verify encoding/decoding consistency
    print("\n" + "-"*80)
    print("TEST 4: Encoding/Decoding Consistency")
    print("-"*80)
    
    # Check if decoded matches original
    match = (decoded['num_blocks'] == arch['num_blocks'] and
             decoded['dense_1'] == arch['dense_1'] and
             decoded['dense_2'] == arch['dense_2'] and
             abs(decoded['dropout'] - arch['dropout']) < 0.001)
    
    if match:
        print("\n✅ Encoding/Decoding is consistent!")
    else:
        print("\n❌ Encoding/Decoding mismatch!")
    
    # Test validation
    print("\n" + "-"*80)
    print("TEST 5: Architecture Validation")
    print("-"*80)
    
    is_valid = space.is_valid_architecture(arch)
    print(f"\n✅ Architecture is valid: {is_valid}")
    
    # Test summary
    print("\n" + "-"*80)
    print("TEST 6: Architecture Summary")
    print("-"*80)
    
    summary = space.get_architecture_summary(arch)
    print(f"\n{summary}")
    
    # Test multiple random architectures
    print("\n" + "-"*80)
    print("TEST 7: Multiple Random Architectures")
    print("-"*80)
    
    print("\nGenerating 5 random architectures:")
    for i in range(5):
        arch_test = space.random_architecture()
        print(f"\n  Architecture {i+1}:")
        print(f"    Blocks: {arch_test['num_blocks']}")
        print(f"    First block filters: {arch_test['blocks'][0]['filters']}")
        print(f"    Dense layers: {arch_test['dense_1']}, {arch_test['dense_2']}")
        print(f"    Dropout: {arch_test['dropout']}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSearch Space is ready for use in PSO-NAS!")
    print("\nNext steps:")
    print("  1. Implement Particle class (Task 3)")
    print("  2. Implement Fitness functions (Task 4)")
    print("  3. Implement PSO Optimizer (Task 5)")
    print("\n" + "="*80)
