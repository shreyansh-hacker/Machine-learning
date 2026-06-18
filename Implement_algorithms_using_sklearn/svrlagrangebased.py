"""
================================================================================
Module Name: svrlagrangebased.py
Description: Support Vector Regression (SVR) using Scikit-Learn.
Methodology:
  1. Read data from 'polynomial_data.csv'.
  2. Fit an SVR model with RBF kernel, custom C penalty, and epsilon margin.
  3. Compute MSE score on training coordinates.
  4. Plot regression fit curve and highlight Support Vectors.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# ============================================================
# Load data
# ============================================================

data = pd.read_csv("polynomial_data.csv")

X = data["x_train"].values.reshape(-1,1)
y = data["y_train"].values

# ============================================================
# Train SVR
# ============================================================

model = SVR(
    kernel='rbf',   # RBF Kernel
    C=100,          # Same as your code
    epsilon=0.3,    # Same epsilon
    gamma=5.0       # Same gamma
)

model.fit(X, y)

# ============================================================
# Prediction on training data
# ============================================================

y_train_pred = model.predict(X)

mse = mean_squared_error(y, y_train_pred)

print("MSE:", mse)

# ============================================================
# Predict smooth curve
# ============================================================

x_plot = np.linspace(
    X.min(),
    X.max(),
    500
).reshape(-1,1)

y_pred = model.predict(x_plot)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.scatter(
    X,
    y,
    color='blue',
    alpha=0.6,
    label='Training Data'
)

plt.plot(
    x_plot,
    y_pred,
    color='red',
    linewidth=3,
    label='SVR Fit'
)

# Support vectors
plt.scatter(
    model.support_vectors_,
    model.predict(model.support_vectors_),
    s=100,
    facecolors='none',
    edgecolors='black',
    label='Support Vectors'
)

plt.legend()
plt.grid(True)
plt.show()