import numpy as np
import matplotlib.pyplot as plt

# Load and inspect your data
X = np.load('features/X.npy')
y = np.load('features/y.npy')

print("DATA DIAGNOSTIC REPORT")
print("=" * 50)

# Basic info
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

# Check for common issues
print(f"LABEL ANALYSIS:")
print(f"Unique labels: {np.unique(y)}")
print(f"Label counts: {np.bincount(y)}")
print(f"Min label: {np.min(y)}, Max label: {np.max(y)}")

# Check data quality
print(f"DATA QUALITY:")
print(f"X min: {np.min(X):.6f}, max: {np.max(X):.6f}")
print(f"X mean: {np.mean(X):.6f}, std: {np.std(X):.6f}")

# Check for problematic values
nan_count = np.sum(np.isnan(X))
inf_count = np.sum(np.isinf(X))
zero_samples = np.sum(np.all(X == 0, axis=(1, 2)))

print(f"NaN values: {nan_count}")
print(f"Inf values: {inf_count}")
print(f"All-zero samples: {zero_samples}")

# Check variance across samples
sample_variances = np.var(X, axis=(1, 2))
print(f"Sample variances - min: {np.min(sample_variances):.6f}, max: {np.max(sample_variances):.6f}")
print(f"Zero variance samples: {np.sum(sample_variances < 1e-10)}")

# Visualize a few samples
if X.shape[0] > 0:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i in range(min(8, X.shape[0])):
        row, col = i // 4, i % 4
        sample = X[i].squeeze()
        axes[row, col].imshow(sample, aspect='auto', cmap='viridis')
        axes[row, col].set_title(f'Sample {i}, Label: {y[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample visualizations saved as 'data_samples.png'")

# Check class balance
print(f"CLASS BALANCE:")
unique_labels, counts = np.unique(y, return_counts=True)
for label, count in zip(unique_labels, counts):
    percentage = (count / len(y)) * 100
    print(f"Class {label}: {count} samples ({percentage:.1f}%)")

imbalance_ratio = np.max(counts) / np.min(counts)
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 5:
    print("WARNING: Severe class imbalance detected!")

# Checks
if nan_count > 0 or inf_count > 0:
    print("Fix NaN/Inf values in your data")
if zero_samples > X.shape[0] * 0.1:
    print("Too many zero samples - check feature extraction")
if np.sum(sample_variances < 1e-10) > X.shape[0] * 0.05:
    print("Many samples have no variance - check preprocessing")
if imbalance_ratio > 3:
    print("Consider using class weights or data balancing")

print("Run this script first to identify data issues!")


# '''
# CLASS BALANCE:
# Class 0: 92 samples (6.7%)
# Class 1: 184 samples (13.3%)
# Class 2: 184 samples (13.3%)
# Class 3: 184 samples (13.3%)
# Class 4: 184 samples (13.3%)
# Class 5: 184 samples (13.3%)
# Class 6: 184 samples (13.3%)
# Class 7: 184 samples (13.3%)
# Imbalance ratio: 2.00
# '''