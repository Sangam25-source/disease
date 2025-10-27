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

# Import main components (Phase 2 complete!)
try:
    from .fuzzy_cnn import (
        FuzzificationLayer,
        FuzzyConvLayer,
        DefuzzificationLayer,
        FixedFuzzyCNN,
        build_fuzzy_cnn
    )
    _FUZZY_CNN_AVAILABLE = True
except ImportError:
    _FUZZY_CNN_AVAILABLE = False

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
