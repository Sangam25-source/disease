# FedED-SegNAS Quick Reference

## 📁 Project Structure

```
FedED-SegNAS/
├── config/experiment_config.yaml  # Main configuration
├── data/
│   ├── simulated/                 # 192 datasets (2.1 GB)
│   ├── processed/                 # Federated splits
│   ├── real/                      # Real genomic data
│   └── raw/                       # Original raw data
├── experiments/
│   ├── generate_simple_datasets.py
│   └── data_preprocessing.py
├── utils/
│   └── data_loader.py
└── models/  [future]
```

---

## 🚀 Common Commands

### Dataset Generation

```bash
cd /app/FedED-SegNAS

# Test mode (2 datasets/config)
python experiments/generate_simple_datasets.py --test

# Full generation (10 datasets/config)
python experiments/generate_simple_datasets.py
```

### Data Preprocessing

```bash
# Preprocess specific number of datasets
python experiments/data_preprocessing.py --limit 10

# Preprocess all datasets
python experiments/data_preprocessing.py

# Custom client configuration
python experiments/data_preprocessing.py --num-clients 100
```

### Data Loading

```python
import numpy as np
import sys
sys.path.append('/app/FedED-SegNAS')
from utils.data_loader import DataLoader

# Scan datasets
loader = DataLoader()
datasets = loader.scan_gametes_datasets()
print(f"Found {len(datasets)} datasets")

# Load raw dataset
X, y = loader.load_gametes_data('data/simulated/model1/order2/snps50/dataset_0.txt')

# Load preprocessed dataset
data = np.load('data/processed/model1/order2/snps50/dataset_0.npz', allow_pickle=True)
X_val = data['validation_X']
y_val = data['validation_y']
```

---

## 📊 Dataset Overview

| Component | Value |
|-----------|-------|
| Total Datasets | 192 |
| Disease Models | 8 |
| Epistasis Orders | 2, 3 |
| SNP Sizes | 50, 100, 500, 1000, 2000, 5000 |
| Samples/Dataset | 4,000 (balanced) |
| Federated Clients | 50 |

---

## 🔑 Key Files

- **config/experiment_config.yaml** - All hyperparameters
- **experiments/generate_simple_datasets.py** - Dataset generation
- **experiments/data_preprocessing.py** - Federated splitting
- **utils/data_loader.py** - Data loading utilities
- **STATUS.md** - Detailed project status
- **DATASET_SUMMARY.md** - Dataset documentation

---

## 🎯 Current Status

✅ Phase 0: Environment Setup  
✅ Phase 1: Dataset Generation & Preprocessing  
🔄 Phase 2: Fuzzy CNN (Next)  
⏳ Phase 3: PSO-NAS  
⏳ Phase 4: Privacy Module  
⏳ Phase 5: Federated Learning  

**Progress:** 20% Complete

---

## 💡 Quick Tips

1. **Start small:** Use `--test` flag for quick iterations
2. **Check disk space:** Full datasets need ~120 GB
3. **Use preprocessed data:** 10x smaller and faster to load
4. **Set PYTHONPATH:** `export PYTHONPATH=/app/FedED-SegNAS:$PYTHONPATH`
5. **Monitor memory:** Large SNP sizes (5000) need more RAM

---

## 🆘 Troubleshooting

**ModuleNotFoundError:**
```bash
export PYTHONPATH=/app/FedED-SegNAS:$PYTHONPATH
```

**Out of disk space:**
```bash
# Check usage
du -sh data/*
# Generate smaller subset with --test
```

**Memory issues:**
```bash
# Process in batches
python experiments/data_preprocessing.py --limit 20
```

---

**Last Updated:** 2025-02-20
