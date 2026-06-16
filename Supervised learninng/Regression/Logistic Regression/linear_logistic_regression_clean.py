import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Reproducible
np.random.seed(42)

# Generate a classification dataset
x, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=42,
)

# Plot dataset
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)

# Simple logistic computations (demo)
w = np.random.rand(1, 2)
b = np.array([[0]])
y = y.reshape(-1, 1)
print(x.shape, y.shape, w.shape)

# Correct sigmoid: sigmoid(z) = 1/(1+exp(-z))
z = w @ x.T + b
y_pred = (1 / (1 + np.exp(-z))).T
print("y_pred shape:", y_pred.shape)

# Cross-entropy cost (vectorized)
num_samples = x.shape[0]
# clip predictions to avoid log(0)
eps = 1e-12
y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
cost = -(1 / num_samples) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
print("cost:", cost)
