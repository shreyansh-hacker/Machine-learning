"""
================================================================================
Module Name: svmlagrangebased.py
Description: Kernel Support Vector Classification using Scikit-Learn's SVC.
Methodology:
  1. Generate a synthetic roasting dataset (temperature vs duration).
  2. Train an SVC model using the Radial Basis Function (RBF) kernel.
  3. Plot decision regions, boundary lines, and highlight Support Vectors.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ================= LOAD COFFEE DATA =================

def load_coffee_data():
    """
    Creates a coffee roasting dataset.
    """
    rng = np.random.default_rng(2)

    X = rng.random(400).reshape(-1, 2)

    # Roasting duration: 11.5 to 15.5 minutes
    X[:, 1] = X[:, 1] * 4 + 11.5

    # Temperature: 150 to 285 degree C
    X[:, 0] = X[:, 0] * (285 - 150) + 150

    Y = np.zeros(len(X))

    for i, (t, d) in enumerate(X):

        y_line = -3/(260 - 175) * t + 21

        if (
            (175 < t < 260) and
            (12 < d < 15) and
            (d <= y_line)
        ):
            Y[i] = 1
        else:
            Y[i] = 0

    return X, Y

# ================= DATA =================

X, y = load_coffee_data()

print("X shape:", X.shape)
print("y shape:", y.shape)

# ================= PLOT DATA =================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    color="red",
    label="Class 0",
    alpha=0.6
)

plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    color="blue",
    label="Class 1",
    alpha=0.6
)

plt.xlabel("Temperature")
plt.ylabel("Duration")
plt.title("Coffee Roasting Dataset")
plt.legend()
plt.grid(True)
plt.show()

# ================= TRAIN SVM =================

model = SVC(
    kernel='rbf',
    C=100.0,
    gamma=0.002
)

model.fit(X, y)

# ================= PREDICTION =================

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print("Accuracy:", accuracy * 100)

# ================= DECISION BOUNDARY =================

x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="coolwarm"
)

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    colors='black',
    linewidths=2
)

plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    color='red',
    label='Class 0'
)

plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    color='blue',
    label='Class 1'
)

# Support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=120,
    facecolors='none',
    edgecolors='black',
    linewidths=1.5,
    label='Support Vectors'
)

plt.xlabel("Temperature")
plt.ylabel("Duration")
plt.title("Kernel SVM (RBF) - sklearn")
plt.legend()
plt.grid(True)
plt.show()