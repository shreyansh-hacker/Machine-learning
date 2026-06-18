"""
================================================================================
Module Name: polynomial_logistic_regression_clean.py
Description: Polynomial Logistic Regression from scratch.
Methodology:
  1. Generate a synthetic 2D binary classification dataset.
  2. Create degree-3 polynomial combinations of input features.
  3. Train a classification model using sigmoid activations and cross-entropy loss gradient descent.
  4. Mesh grid predictions to plot the non-linear decision boundary boundary line.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Reproducible
np.random.seed(42)

# Generate a simple 2D classification dataset
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

print("Features shape:", x.shape)
print("Target shape:", y.shape)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dataset for Polynomial Logistic Regression Demo")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Create polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x)
print("Polynomial features shape:", x_poly.shape)

# Fit a logistic regression on polynomial features (quick demo)
model = LogisticRegression(max_iter=1000)
model.fit(x_poly, y)
acc = model.score(x_poly, y)
print("Training accuracy (polynomial features):", acc)
num_samples, num_features = x_poly.shape

w = np.zeros((1, num_features))
b = 0
learning_rate = 0.
iterations = 1500
cost_history = []
for i in range(iterations):
    # 1. Prediction (Linear Combination + Sigmoid)
    z = w @ x_poly.T + b
    y_pred = 1 / (1 + np.exp(-z)) # Shape: (1, 200)
    
    # 2. Error
    error = y_pred - y.T # Shape: (1, 200)
    
    # 3. Gradients (Matrix Transpose Logic)
    djdw = (1 / num_samples) * (error @ x_poly) # Shape: (1, 9)
    djdb = (1 / num_samples) * np.sum(error)
    
    # 4. Weights aur Bias Update
    w = w - learning_rate * djdw
    b = b - learning_rate * djdb
    
    # 5. Cost calculate karke save karna
    cost = -(1 / num_samples) * np.sum(y.T * np.log(y_pred) + (1 - y.T) * np.log(1 - y_pred))
    cost_history.append(cost)

# Grid banakar curve region plot karna
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Grid points ko bhi degree 3 polynomial mein badalna padega prediction ke liye
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_poly = poly.transform(grid_points) # Sklearn poly object ka use kiya

# Manual weights (w, b) se predict karna
z_grid = w @ grid_poly.T + b
preds_grid = 1 / (1 + np.exp(-z_grid))
preds_grid = preds_grid.reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, preds_grid, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
plt.contour(xx, yy, preds_grid, levels=[0.5], colors='black', linewidths=2) 

plt.scatter(x[y == 0, 0], x[y == 0, 1], color="red", alpha=0.6)
plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", alpha=0.6)
plt.show()