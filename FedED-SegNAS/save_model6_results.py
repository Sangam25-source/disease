#!/usr/bin/env python3
"""
Save Model6 results that were already obtained.
Run this to save results without retraining.
"""
import os
import json
import csv
from datetime import datetime

# Your results from the previous run
results = [
    {'order': 2, 'snp_size': 50, 'dataset_id': 0, 'test_accuracy': 0.6450},
    {'order': 2, 'snp_size': 50, 'dataset_id': 1, 'test_accuracy': 0.6433},
    {'order': 2, 'snp_size': 100, 'dataset_id': 0, 'test_accuracy': 0.6200},
    {'order': 2, 'snp_size': 100, 'dataset_id': 1, 'test_accuracy': 0.6217},
    {'order': 2, 'snp_size': 500, 'dataset_id': 0, 'test_accuracy': 0.6233},
    {'order': 2, 'snp_size': 500, 'dataset_id': 1, 'test_accuracy': 0.6117},
    {'order': 2, 'snp_size': 1000, 'dataset_id': 0, 'test_accuracy': 0.6200},
    {'order': 2, 'snp_size': 1000, 'dataset_id': 1, 'test_accuracy': 0.6500},
    {'order': 2, 'snp_size': 2000, 'dataset_id': 0, 'test_accuracy': 0.6333},
    {'order': 2, 'snp_size': 2000, 'dataset_id': 1, 'test_accuracy': 0.6167},
    {'order': 2, 'snp_size': 5000, 'dataset_id': 0, 'test_accuracy': 0.6083},
    {'order': 2, 'snp_size': 5000, 'dataset_id': 1, 'test_accuracy': 0.6067},
    {'order': 3, 'snp_size': 50, 'dataset_id': 0, 'test_accuracy': 0.5183},
    {'order': 3, 'snp_size': 50, 'dataset_id': 1, 'test_accuracy': 0.5383},
    {'order': 3, 'snp_size': 100, 'dataset_id': 0, 'test_accuracy': 0.5167},
    {'order': 3, 'snp_size': 100, 'dataset_id': 1, 'test_accuracy': 0.5350},
    {'order': 3, 'snp_size': 500, 'dataset_id': 0, 'test_accuracy': 0.5333},
    {'order': 3, 'snp_size': 500, 'dataset_id': 1, 'test_accuracy': 0.5167},
    {'order': 3, 'snp_size': 1000, 'dataset_id': 0, 'test_accuracy': 0.5233},
    {'order': 3, 'snp_size': 1000, 'dataset_id': 1, 'test_accuracy': 0.5467},
    {'order': 3, 'snp_size': 2000, 'dataset_id': 0, 'test_accuracy': 0.4883},
    {'order': 3, 'snp_size': 2000, 'dataset_id': 1, 'test_accuracy': 0.5000},
    {'order': 3, 'snp_size': 5000, 'dataset_id': 0, 'test_accuracy': 0.5283},
    {'order': 3, 'snp_size': 5000, 'dataset_id': 1, 'test_accuracy': 0.4833},
]

# Add model info to each result
for r in results:
    r['model_name'] = 'model6'
    r['model_type'] = 'Pure Epistasis'

# Create output directory
output_dir = 'results/model6_results'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save to CSV
csv_path = os.path.join(output_dir, f'model6_results_{timestamp}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model_name', 'model_type', 'order', 'snp_size', 'dataset_id', 'test_accuracy'])
    writer.writeheader()
    writer.writerows(results)
print(f"CSV saved: {csv_path}")

# Save to JSON
json_path = os.path.join(output_dir, f'model6_results_{timestamp}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"JSON saved: {json_path}")

# Calculate statistics
avg_acc = sum(r['test_accuracy'] for r in results) / len(results)
best_acc = max(r['test_accuracy'] for r in results)
worst_acc = min(r['test_accuracy'] for r in results)

order2_results = [r for r in results if r['order'] == 2]
order3_results = [r for r in results if r['order'] == 3]
order2_avg = sum(r['test_accuracy'] for r in order2_results) / len(order2_results)
order3_avg = sum(r['test_accuracy'] for r in order3_results) / len(order3_results)

# Save summary text
summary_path = os.path.join(output_dir, f'model6_summary_{timestamp}.txt')
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL6 EVALUATION RESULTS SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model Type: Pure Epistasis\n\n")
    
    f.write(f"Total datasets tested: {len(results)}\n")
    f.write(f"Average accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")
    f.write(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    f.write(f"Worst accuracy: {worst_acc:.4f} ({worst_acc*100:.2f}%)\n\n")
    
    f.write(f"Order 2 average: {order2_avg:.4f} ({order2_avg*100:.2f}%)\n")
    f.write(f"Order 3 average: {order3_avg:.4f} ({order3_avg*100:.2f}%)\n")
    
    f.write("\n" + "-"*70 + "\n")
    f.write("Per-dataset results:\n")
    f.write(f"{'Dataset':<45} {'Accuracy':<15}\n")
    f.write("-"*70 + "\n")
    for r in results:
        name = f"order{r['order']}/snps{r['snp_size']}/dataset_{r['dataset_id']}"
        status = "GOOD" if r['test_accuracy'] >= 0.60 else ""
        f.write(f"{name:<45} {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%) {status}\n")
    
    f.write("\n" + "-"*70 + "\n")
    f.write("Note: Model6 is PURE EPISTASIS - individual SNPs have\n")
    f.write("no predictive power. 60-65% accuracy is the expected\n")
    f.write("ceiling for this type of data.\n")
    f.write("-"*70 + "\n")

print(f"Summary saved: {summary_path}")

print("\n" + "="*50)
print("MODEL6 RESULTS SAVED SUCCESSFULLY!")
print("="*50)
print(f"Files saved to: {output_dir}/")
print(f"  - {os.path.basename(csv_path)}")
print(f"  - {os.path.basename(json_path)}")
print(f"  - {os.path.basename(summary_path)}")
print("="*50)
