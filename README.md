# Here are your Instructions
# Train all models with default settings
python train.py

# High accuracy mode (slower but better)
python train.py --high-accuracy

# Quick test
python train.py --quick

# Train specific model
python train.py --model model1

# Custom configuration
python train.py --min-rounds 100 --max-rounds 300 --local-epochs 2

# Enable NAS
python train.py --with-nas
