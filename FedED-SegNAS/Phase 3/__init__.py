"""
Phase 3: Federated Learning Module
===================================

This module implements federated learning for FedED-SegNAS project.

Components:
-----------
- SimpleFederatedTrainer: Main federated learning trainer class
- train_federated.py: Training script for federated learning
- test_federated_learning.py: Unit tests for federated learning

Usage:
------
# Import the trainer
from Phase_3 import SimpleFederatedTrainer

# Or import from federated_learning module
from Phase_3.federated_learning import SimpleFederatedTrainer

# Initialize trainer
trainer = SimpleFederatedTrainer(num_snps=50, num_clients=50)

# Train on federated data
history = trainer.train(federated_data, num_rounds=50)

# Evaluate
results = trainer.evaluate(X_test, y_test)

Author: FedED-SegNAS Project
Date: January 2026
"""

# Import main class
from .federated_learning import SimpleFederatedTrainer

# Version info
__version__ = '1.0.0'
__author__ = 'FedED-SegNAS Project'

# Public API
__all__ = [
    'SimpleFederatedTrainer',
]

# Module metadata
__doc_url__ = 'https://github.com/your-repo/FedED-SegNAS'
__description__ = 'Federated Learning Framework for Epistasis Detection'
