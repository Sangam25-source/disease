# FedED-SegNAS: Federated Epistasis Detection with Segmented Neural Architecture Search

## Project Overview
Implementation of FedED-SegNAS framework for epistasis detection in genomic data using:
- Fuzzy CNN for feature extraction
- PSO-based Neural Architecture Search
- Privacy-preserving federated learning
- Sequence perturbation for secure communication

## Project Structure
```
FedED-SegNAS/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed data
│   ├── simulated/           # GAMETES generated data
│   └── real/                # RA, AMD datasets
├── models/
│   ├── fuzzy_cnn.py        # Fuzzy CNN implementation
│   ├── pso_nas.py          # PSO-based NAS
│   ├── privacy.py          # Sequence perturbation
│   └── federated_learning.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   └── config_parser.py
├── experiments/
│   ├── train_federated.py
│   ├── evaluate.py
│   └── ablation_study.py
├── results/
│   ├── logs/
│   ├── plots/
│   └── metrics/
├── notebooks/
├── config/
└── requirements.txt
```

## Installation

### 1. Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. External Tools
- Java (for GAMETES)
- PLINK (for genomic data processing)

## Usage

### Phase 0: Dataset Generation
```bash
# Download GAMETES
cd data/simulated
wget https://sourceforge.net/projects/gametes/files/GAMETES_2.0.jar/download -O GAMETES_2.0.jar

# Generate datasets
python experiments/generate_datasets.py
```

### Phase 1: Data Preprocessing
```bash
python experiments/data_preprocessing.py
```

### Phase 2-3: Model Training
```bash
python experiments/train_federated.py --config config/experiment_config.yaml
```

## Dataset Specifications

### Simulated Datasets (GAMETES)
- 8 disease models (varying heritability, MAF, epistatic patterns)
- 6 SNP sizes: [50, 100, 500, 1000, 2000, 5000]
- 100 datasets per configuration
- Total: 4,800 datasets

### Real Datasets
- Rheumatoid Arthritis (RA)
- Age-related Macular Degeneration (AMD)
- 1000 Genomes Project (for testing)

## License
Academic Research Use

## Citation
If you use this code, please cite the original FedED-SegNAS paper.