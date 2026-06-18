"""
================================================================================
Module Name: logisticregression.py
Description: Logistic Regression using Scikit-Learn.
Methodology:
  1. Generate classification dataset.
  2. Scale inputs using StandardScaler.
  3. Train a LogisticRegression model and retrieve classification accuracy.
  4. Solve parameters analytically to plot the linear decision boundary line.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Generate Dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_scaled, y)

# Accuracy
accuracy = model.score(X_scaled, y)*100
print("Accuracy:", accuracy)

# Plot Data
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[y==0][:,0], X_scaled[y==0][:,1],
            color='red', label='Class 0', alpha=0.6)

plt.scatter(X_scaled[y==1][:,0], X_scaled[y==1][:,1],
            color='blue', label='Class 1', alpha=0.6)

# Decision Boundary
w = model.coef_[0]
b = model.intercept_[0]

x1 = np.linspace(X_scaled[:,0].min(),
                 X_scaled[:,0].max(), 100)

x2 = -(w[0]*x1 + b)/w[1]

plt.plot(x1, x2, color='black', label='Decision Boundary')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()