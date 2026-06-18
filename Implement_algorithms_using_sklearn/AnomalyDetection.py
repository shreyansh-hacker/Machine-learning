"""
================================================================================
Module Name: AnomalyDetection.py
Description: Anomaly Detection using Scikit-Learn's EllipticEnvelope.
Methodology:
  1. Load a synthetic coffee dataset containing temperature and roasting duration.
  2. Label normal points using mathematical roasting limits (Normal vs Anomaly).
  3. Standardize and split the data, filtering out anomalies for training.
  4. Fit an EllipticEnvelope model only on the normal roasting dataset.
  5. Predict anomalies on the full dataset and evaluate using Accuracy and F1 Score.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score

# ==========================================================
# Load Coffee Data
# ==========================================================

def load_coffee_data():

    rng = np.random.default_rng(2)

    X = rng.random(400).reshape(-1, 2)

    # Duration: 11.5 to 15.5 minutes
    X[:, 1] = X[:, 1] * 4 + 11.5

    # Temperature: 150 to 285 C
    X[:, 0] = X[:, 0] * (285 - 150) + 150

    Y = np.zeros(len(X))

    for i, (t, d) in enumerate(X):

        y_line = -3/(260 - 175) * t + 21

        if (
            175 < t < 260 and
            12 < d < 15 and
            d <= y_line
        ):
            Y[i] = 1   # Normal
        else:
            Y[i] = 0   # Anomaly

    return X, Y

# ==========================================================
# Load Data
# ==========================================================

X, y = load_coffee_data()

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================================
# Plot Original Data
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    color="red",
    label="Anomaly",
    alpha=0.6
)

plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    color="blue",
    label="Normal",
    alpha=0.6
)

plt.xlabel("Temperature")
plt.ylabel("Duration")
plt.title("Original Coffee Dataset")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================================
# Train only on Normal Data
# ==========================================================

normal_data = X[y == 1]

model = EllipticEnvelope(
    contamination=np.mean(y == 0),
    random_state=42
)

model.fit(normal_data)

# ==========================================================
# Prediction
# ==========================================================

pred = model.predict(X)

# sklearn:
#  1  -> normal
# -1 -> anomaly

pred = np.where(pred == 1, 1, 0)

accuracy = np.mean(pred == y) * 100
print("Accuracy:", accuracy)

f1 = f1_score(y, pred)
print("F1 Score:", f1)

# ==========================================================
# Plot Predictions
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[pred == 0][:, 0],
    X[pred == 0][:, 1],
    color="red",
    label="Predicted Anomaly",
    alpha=0.6
)

plt.scatter(
    X[pred == 1][:, 0],
    X[pred == 1][:, 1],
    color="blue",
    label="Predicted Normal",
    alpha=0.6
)

plt.xlabel("Temperature")
plt.ylabel("Duration")
plt.title("Anomaly Detection using sklearn")
plt.legend()
plt.grid(True)

plt.show()