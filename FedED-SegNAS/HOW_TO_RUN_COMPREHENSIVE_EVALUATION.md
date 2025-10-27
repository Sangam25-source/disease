# 📋 Instructions: Running Comprehensive Model Evaluation

## 🎯 Purpose
This script will train and evaluate the Fuzzy CNN on **ALL 8 disease models** for **30 epochs** to get accurate performance metrics.

---

## 📦 Prerequisites

Make sure you have:
1. ✅ Python 3.8+ installed
2. ✅ All dependencies installed (TensorFlow, NumPy, Pandas, Matplotlib, Seaborn)
3. ✅ All preprocessed data in `data/processed/` directory

---

## 🚀 How to Run

### Option 1: Default Run (Recommended)
```bash
cd /app/FedED-SegNAS
python3 experiments/comprehensive_model_evaluation.py
```

**This will:**
- Test all 8 models (model1-8)
- Use 30 epochs (default)
- Test on 50 SNPs (default)
- Train on all available datasets per model

**Expected Duration:** ~15-25 minutes total (depending on your CPU)

---

### Option 2: Custom Epochs
```bash
cd /app/FedED-SegNAS
python3 experiments/comprehensive_model_evaluation.py --epochs 30
```

---

### Option 3: Test Different SNP Size
```bash
# For 100 SNPs
python3 experiments/comprehensive_model_evaluation.py --snps 100 --epochs 30

# For 500 SNPs
python3 experiments/comprehensive_model_evaluation.py --snps 500 --epochs 30
```

---

### Option 4: Quick Test (For Testing Script)
```bash
# This tests only 1 dataset per model with fewer epochs
python3 experiments/comprehensive_model_evaluation.py --quick --epochs 10
```

---

## 📊 What You'll See

### During Execution:
```
================================================================================
🧬 COMPREHENSIVE FUZZY CNN EVALUATION: ALL 8 DISEASE MODELS
================================================================================
📅 Start Time: 2024-10-27 12:30:45
⚙️  Configuration:
   - Epochs: 30
   - SNP Size: 50
   - Quick Mode: False
================================================================================

================================================================================
🔬 TESTING MODEL1
================================================================================
Type: Marginal
Order: 2-way epistasis
Expected Accuracy: 85-95%
--------------------------------------------------------------------------------
📁 Found 4 dataset(s) with 50 SNPs

────────────────────────────────────────────────────────────────────────────────
📊 Dataset 1/4: order2/snps50/dataset_0
────────────────────────────────────────────────────────────────────────────────

  📊 Loading data from: dataset_0.npz
  📈 Data: Train=2800, Val=600, Test=600, SNPs=50
  🔧 Model: 259,336 parameters
  🏃 Training for 30 epochs...
  
  Epoch 1/30
  44/44 ━━━━━━━━━━━━━━━━━━━━ 2s 28ms/step - accuracy: 0.5125 - loss: 0.6943
  Epoch 2/30
  44/44 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.5489 - loss: 0.6872
  ...
  Epoch 30/30
  44/44 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - accuracy: 0.9125 - loss: 0.2543
  
  ✅ Complete!
     Test Accuracy: 0.8950 (89.50%)
     Test Loss: 0.2834
     Training Time: 0.82 minutes
     Performance: Excellent

[... continues for all 8 models ...]
```

---

## 📁 Output Files

After completion, you'll find these files in `results/comprehensive_evaluation/`:

### 1. Results CSV
**File:** `results_YYYYMMDD_HHMMSS.csv`

Contains columns:
- model_name
- model_type
- epistasis_order
- num_snps
- epochs
- test_accuracy
- test_loss
- final_train_acc
- final_val_acc
- training_time_minutes
- performance_category
- expected_accuracy

### 2. Results JSON
**File:** `results_YYYYMMDD_HHMMSS.json`

Full detailed results in JSON format.

### 3. Summary Report
**File:** `summary_YYYYMMDD_HHMMSS.txt`

Text summary with:
- Overall statistics
- Per-model results
- Performance categories

### 4. Visualizations
**Files:**
- `accuracy_comparison_YYYYMMDD_HHMMSS.png` - Bar chart of accuracy by model
- `training_time_YYYYMMDD_HHMMSS.png` - Training time comparison

---

## 📋 What to Share Back

After the script completes, please share:

### 1. Terminal Output
Copy the final summary section (looks like this):
```
================================================================================
📊 SUMMARY TABLE
================================================================================
[... table with results ...]

================================================================================
✅ COMPREHENSIVE EVALUATION COMPLETE
================================================================================
Models Tested: 8/8
Total Datasets: 28
Average Accuracy: 0.8542 (85.42%)
Results Directory: results/comprehensive_evaluation/
```

### 2. CSV File
Share the `results_YYYYMMDD_HHMMSS.csv` file

### 3. Summary Text File
Share the `summary_YYYYMMDD_HHMMSS.txt` file

### 4. Screenshots (Optional)
Screenshots of the two PNG visualization files

---

## ⚠️ Troubleshooting

### Issue: "No datasets found"
**Solution:** Check if data is in the correct location:
```bash
ls -la data/processed/model1/order2/snps50/
```

### Issue: "Out of Memory"
**Solution:** Reduce batch size in the script:
- Edit line with `batch_size=64` to `batch_size=32`

### Issue: Script takes too long
**Solution:** Use quick mode first:
```bash
python3 experiments/comprehensive_model_evaluation.py --quick --epochs 10
```

### Issue: ModuleNotFoundError
**Solution:** Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## ⏱️ Expected Timing

For **30 epochs** on **50 SNPs**:

| Component | Time per Dataset | Total (8 models × 4 datasets) |
|-----------|------------------|-------------------------------|
| Single dataset | ~0.8-1.2 minutes | ~25-40 minutes |

**Note:** Actual time varies based on your CPU/GPU.

---

## 📊 Expected Results

Based on epistasis literature:

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| model1 | Marginal | 85-95% |
| model2 | Marginal | 85-95% |
| model3 | Marginal | 85-95% |
| model4 | 3-way Mixed | 75-88% |
| model5 | Pure Epistasis | 70-85% |
| model6 | Pure Epistasis | 70-85% |
| model7 | Marginal | 85-95% |
| model8 | Marginal | 85-95% |

**Overall Expected Average:** 80-90%

---

## 🎯 Success Criteria

Your run is successful if:
- ✅ All 8 models are tested (or 7 if model4 doesn't have 50 SNPs)
- ✅ Test accuracy > 70% for at least 6 models
- ✅ Average accuracy > 75%
- ✅ CSV and summary files are generated
- ✅ No errors in execution

---

## 📞 If You Need Help

If you encounter any issues:
1. Check the troubleshooting section above
2. Share the error message
3. Share the terminal output up to the point of failure

---

## 🎉 After Completion

Once you have the results, share them back and I will:
1. ✅ Analyze the performance across all models
2. ✅ Update the Phase 2 report with actual results
3. ✅ Create detailed performance analysis
4. ✅ Compare against expected baselines
5. ✅ Identify which models need further tuning

---

**Ready to run? Execute:**
```bash
cd /app/FedED-SegNAS
python3 experiments/comprehensive_model_evaluation.py
```

**Good luck! 🚀**
