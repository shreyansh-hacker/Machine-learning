"""
================================================================================
Module Name: polynomial_data.py
Description: Polynomial Regression Pipeline using Scikit-Learn.
Methodology:
  1. Read x_train and y_train from 'polynomial_data.csv'.
  2. Define pipeline with PolynomialFeatures (degree=17), StandardScaler, and LinearRegression.
  3. Fit the model and compute MSE and R2 scores.
  4. Plot data points against the fitted high-degree polynomial curve.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("D:\\ML using Scikit-Learn\\polynomial_data.csv")

X = data["x_train"].values.reshape(-1, 1)
y = data["y_train"].values

# Degree of polynomial
d = 17

# Create pipeline
model = Pipeline([
    ('poly', PolynomialFeatures(degree=d, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE :", mse)
print("R² Score :", r2)

# Plot data and prediction
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="blue", label="Training Data", alpha=0.7)

# Sort for smooth curve
idx = np.argsort(X[:,0])

plt.plot(
    X[idx],
    y_pred[idx],
    color="red",
    linewidth=2,
    label="Polynomial Regression"
)

plt.title("Polynomial Regression using Scikit-Learn")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", X.shape)
print("Target shape is:", y.shape)