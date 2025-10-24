"""
FedED-SegNAS Models Module

This module contains the core model implementations for the FedED-SegNAS framework:

1. Fuzzy CNN (fuzzy_cnn.py):
   - Fuzzification layers
   - Fuzzy convolutional layers
   - Fuzzy pooling layers
   - Defuzzification layers
   - Complete Fuzzy CNN architecture

2. Federated Learning (federated_learning.py):
   - Client training logic
   - Server aggregation (FedAvg)
   - Communication protocols
   - Model synchronization

Note: Privacy module removed as per project requirements.
"""

__version__ = "0.1.0"
__author__ = "FedED-SegNAS Team"

# Import main components (will be available after Phase 2 implementation)
try:
    from .fuzzy_cnn import (
        FuzzificationLayer,
        FuzzyConvLayer,
        FuzzyPoolingLayer,
        DefuzzificationLayer,
        FixedFuzzyCNN,
        build_fuzzy_cnn
    )
except ImportError:
    # Modules not yet implemented
    pass

try:
    from .federated_learning import (
        SimpleFederatedTrainer
    )
except ImportError:
    # Module not yet implemented
    pass

__all__ = [
    'FuzzificationLayer',
    'FuzzyConvLayer',
    'FuzzyPoolingLayer',
    'DefuzzificationLayer',
    'FixedFuzzyCNN',
    'build_fuzzy_cnn',
    'SimpleFederatedTrainer'
]
