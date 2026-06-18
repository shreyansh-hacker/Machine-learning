"""
================================================================================
Module Name: KNN_LogisticRegression.py
Description: Classification using Scikit-Learn's KNeighborsClassifier.
Methodology:
  1. Generate a synthetic 2D binary classification dataset.
  2. Define and train a KNeighborsClassifier with k=5.
  3. Evaluate training accuracy score.
  4. Plot class predictions using distinct colors.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set seed
np.random.seed(42)

# Generate dataset
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

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ==========================
# KNN Classifier
# ==========================

k = 5

model = KNeighborsClassifier(
    n_neighbors=k
)

# Train model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred) * 100
print("Accuracy:", accuracy)

# ==========================
# Plot Predictions
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(
    X[y_pred == 0][:,0],
    X[y_pred == 0][:,1],
    color="red",
    label="Predicted Class 0",
    alpha=0.6
)

plt.scatter(
    X[y_pred == 1][:,0],
    X[y_pred == 1][:,1],
    color="blue",
    label="Predicted Class 1",
    alpha=0.6
)

plt.title("KNN Classification (sklearn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()