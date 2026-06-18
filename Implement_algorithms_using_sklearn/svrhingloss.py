"""
================================================================================
Module Name: svrhingloss.py
Description: High-Degree Polynomial Regression using Scikit-Learn.
Methodology:
  1. Load polynomial coordinates from CSV.
  2. Create degree-8 polynomial pipeline with standard linear regression.
  3. Compute MSE and plot the fitted curve.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("polynomial_data.csv")

X = data["x_train"].values.reshape(-1,1)
y = data["y_train"].values

# Polynomial Regression
degree = 8

model = make_pipeline(
    PolynomialFeatures(degree=degree),
    LinearRegression()
)

model.fit(X, y)

# Predict smooth curve
x_plot = np.linspace(
    X.min(),
    X.max(),
    500
).reshape(-1,1)

y_plot = model.predict(x_plot)

# Training prediction
y_pred = model.predict(X)

print("MSE:", mean_squared_error(y, y_pred))

# Plot
plt.figure(figsize=(10,6))

plt.scatter(X, y, color='blue', label='Training Data')

plt.plot(
    x_plot,
    y_plot,
    color='red',
    linewidth=3,
    label=f'Polynomial Degree {degree}'
)

plt.legend()
plt.grid(True)
plt.show()