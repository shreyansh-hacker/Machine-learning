"""
================================================================================
Module Name: pca.py
Description: Manual implementation of Principal Component Analysis (PCA) using NumPy.
Methodology:
  1. Standardize/Center data by subtracting the mean of each feature.
  2. Compute the Covariance Matrix of the centered features.
  3. Calculate eigenvalues and eigenvectors via eigendecomposition.
  4. Sort eigenvectors by eigenvalues in descending order.
  5. Project the dataset onto the top k eigenvectors (Principal Components).
  6. Reconstruct the original 3D coordinates from the 2D projected coordinates.
Visualization:
  - 2D scatter plot of the data projected onto PC1 and PC2.
  - 3D comparison plot of original centered data vs reconstructed data.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Step 0: Create Dummy Data ---
np.random.seed(0)
X = np.random.randn(100, 3)  # 100 samples, 3 features

# --- Step 1: Standardize ---
X_centered = X - np.mean(X, axis=0)

# --- Step 2: Covariance Matrix ---
cov = np.cov(X_centered.T)

# --- Step 3: Eigen Decomposition ---
eig_vals, eig_vecs = np.linalg.eig(cov)

# --- Step 4: Sort Eigenvectors by Eigenvalues ---
sorted_idx = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, sorted_idx]

# --- Step 5: Project to Lower Dimensions (e.g., 2D) ---
k = 2
W = eig_vecs[:, :k]
X_reduced = X_centered @ W
X_reconstructed = X_reduced@W.T

# --- Plot Result ---
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
plt.title("PCA Projection to 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.axis('equal')
plt.show()

# -----------------------------
# Plot Original vs Reconstructed
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_centered[:, 0],
    X_centered[:, 1],
    X_centered[:, 2],
    label='Original Centered Data',
    alpha=0.6
)

ax.scatter(
    X_reconstructed[:, 0],
    X_reconstructed[:, 1],
    X_reconstructed[:, 2],
    label='Reconstructed Data',
    alpha=0.6
)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Original vs PCA Reconstructed Data')
ax.legend()

plt.show()