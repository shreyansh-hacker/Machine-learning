"""
================================================================================
Module Name: svmhinghloss.py
Description: Polynomial Support Vector Classification using Scikit-Learn's SVC.
Methodology:
  1. Set up a pipeline containing StandardScaler, PolynomialFeatures (degree=8), and SVC (linear kernel).
  2. Fit the model on training data.
  3. Plot non-linear decision boundaries, margins, and highlight Support Vectors.
================================================================================
"""

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=8, include_bias=False)),
    ("svm", SVC(
        kernel='linear',
        C=10
    ))
])

model.fit(X, y)

# ================= Decision Boundary =================

x_min, x_max = X[:,0].min()-5, X[:,0].max()+5
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Decision function values
Z = model.decision_function(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

# Decision boundary and margins
plt.contour(
    xx,
    yy,
    Z,
    levels=[-1, 0, 1],   # support vectors, hyperplane, support vectors
    colors=['green', 'black', 'green'],
    linestyles=['--', '-', '--'],
    linewidths=2
)

# Data points
plt.scatter(
    X[y==0][:,0],
    X[y==0][:,1],
    color='blue',
    label='Class 0'
)

plt.scatter(
    X[y==1][:,0],
    X[y==1][:,1],
    color='red',
    label='Class 1'
)

# Support vectors
sv = model.named_steps['svm'].support_vectors_

plt.scatter(
    sv[:,0],
    sv[:,1],
    s=150,
    facecolors='none',
    edgecolors='black',
    linewidths=2,
    label='Support Vectors'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Polynomial Features + Linear SVM")
plt.legend()
plt.grid(True)
plt.show()