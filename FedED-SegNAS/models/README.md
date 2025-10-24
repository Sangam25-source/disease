# Models Directory

## Overview
This directory contains the core machine learning models for the FedED-SegNAS framework.

## Structure

```
models/
├── __init__.py                # Module initialization
├── fuzzy_cnn.py              # Fuzzy CNN implementation (Phase 2)
├── federated_learning.py     # Federated learning framework (Phase 3)
└── README.md                 # This file
```

## Components

### 1. Fuzzy CNN (`fuzzy_cnn.py`) - Phase 2

**Status:** 🟡 To be implemented

**Purpose:** Implements the Fuzzy Convolutional Neural Network for epistasis detection.

**Key Classes:**
- `FuzzificationLayer`: Converts SNP data to fuzzy sets using Gaussian membership functions
- `FuzzyConvLayer`: Convolutional layer with fuzzy weights
- `FuzzyPoolingLayer`: Pooling layer for fuzzy features
- `DefuzzificationLayer`: Converts fuzzy outputs back to crisp values
- `FixedFuzzyCNN`: Complete Fuzzy CNN architecture (no NAS)
- `build_fuzzy_cnn()`: Factory function to build and compile the model

**Architecture:**
```
Input (SNPs)
    ↓
Fuzzification Layer (Gaussian membership)
    ↓
Fuzzy Conv Block 1 (64 filters) → Pool → Dropout
    ↓
Fuzzy Conv Block 2 (128 filters) → Pool → Dropout
    ↓
Fuzzy Conv Block 3 (256 filters) → Pool
    ↓
Defuzzification Layer
    ↓
Flatten
    ↓
Dense (512) → Dropout
    ↓
Dense (256) → Dropout
    ↓
Output (2 classes: case/control)
```

**Usage Example:**
```python
from models.fuzzy_cnn import build_fuzzy_cnn
import numpy as np

# Build model
model = build_fuzzy_cnn(num_snps=50, num_classes=2, learning_rate=0.001)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_test)
```

---

### 2. Federated Learning (`federated_learning.py`) - Phase 3

**Status:** 🟡 To be implemented

**Purpose:** Implements federated learning framework for distributed training.

**Key Classes:**
- `SimpleFederatedTrainer`: Main federated learning coordinator
  - `train_client()`: Train model on single client's data
  - `aggregate_weights()`: FedAvg algorithm implementation
  - `train()`: Main federated training loop
  - `evaluate()`: Evaluate global model

**Federated Learning Process:**
```
1. Initialize global model on server
2. For each communication round:
   a. Select random subset of clients (e.g., 10 out of 50)
   b. Send global model to selected clients
   c. Clients train on local data (5 epochs)
   d. Clients send updated weights back to server
   e. Server aggregates weights using FedAvg
   f. Update global model
   g. Evaluate on validation set
3. Final evaluation on test set
```

**Usage Example:**
```python
from models.federated_learning import SimpleFederatedTrainer
import numpy as np

# Load federated data
data = np.load('data/processed/model1/order2/snps50/dataset_0.npz', allow_pickle=True)

# Initialize trainer
trainer = SimpleFederatedTrainer(
    num_snps=50,
    num_clients=50,
    learning_rate=0.001
)

# Train
history = trainer.train(
    federated_data=data,
    num_rounds=50,
    clients_per_round=10,
    local_epochs=5
)

# Evaluate
X_test, y_test = data['test']
results = trainer.evaluate(X_test, y_test)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

---

## Implementation Timeline

| Phase | Component | Status | Duration |
|-------|-----------|--------|----------|
| Phase 0-1 | Data & Setup | ✅ Complete | Week 1-2 |
| **Phase 2** | **Fuzzy CNN** | 🟡 **Next** | **Week 3** |
| Phase 3 | Federated Learning | ⏳ Pending | Week 4 |
| Phase 4 | Evaluation | ⏳ Pending | Week 5 |
| Phase 5 | Documentation | ⏳ Pending | Week 6 |
| Phase 6 | Demo System | ⏳ Pending | Week 6 |

---

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints where possible
- Document all classes and functions with docstrings
- Add inline comments for complex logic

### Testing
- Create unit tests for each layer
- Test with small datasets first
- Verify output shapes at each layer
- Check gradient flow

### Performance Considerations
- Use TensorFlow GPU acceleration when available
- Implement efficient data loading
- Monitor memory usage during training
- Use mixed precision training if needed

---

## Dependencies

All models require:
```python
tensorflow>=2.10.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

See `requirements.txt` in the project root for complete dependencies.

---

## Notes

### Privacy Module Status
The original FedED-SegNAS paper includes a sequence perturbation privacy-preserving mechanism. However, for this college project implementation, the privacy module has been **intentionally removed** to:
- Simplify implementation
- Reduce computational overhead
- Focus on core machine learning functionality
- Make the demo more accessible

The federated learning framework (`SimpleFederatedTrainer`) implements standard FedAvg without additional privacy mechanisms.

### NAS Module Status
The original paper includes Neural Architecture Search (NAS) using Particle Swarm Optimization (PSO). For this implementation, we use a **fixed architecture** to:
- Avoid expensive architecture search (would take weeks)
- Enable training on CPU/modest GPU
- Focus on proving the federated learning concept
- Simplify demonstration for academic purposes

The architecture is pre-defined in `FixedFuzzyCNN` based on optimal configurations from the paper.

---

## Next Steps

### Immediate (Phase 2):
1. Implement `fuzzy_cnn.py` with all layers
2. Create `test_fuzzy_cnn.py` to validate implementation
3. Train on single dataset to verify functionality
4. Document architecture decisions

### Upcoming (Phase 3):
1. Implement `federated_learning.py`
2. Test with multiple clients
3. Verify convergence
4. Measure communication overhead

---

**Last Updated:** Current Session  
**Status:** Directory structure created, ready for Phase 2 implementation
