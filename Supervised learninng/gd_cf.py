"""
================================================================================
Module Name: gd_cf.py
Description: Vectorized Simple Linear Regression with Gradient Descent.
Methodology:
  1. Generate synthetic 1D linear data.
  2. Compute predictions, cost (MSE), and parameter gradients using NumPy vectorization.
  3. Update weights using gradient descent rules.
  4. Plot optimized regression fit line and cost history.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, 1)

# Generate corresponding y values with a linear relationship (y = 4 + 3x + noise)
true_slope = 3
true_intercept = 4
noise = np.random.randn(num_samples, 1)

y_train = true_intercept + true_slope * x_train + noise

# Plotting the data
plt.figure(figsize = (8, 5))
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)

w = 0
b = 0 
learning_rate = 0.01
trials = 1000

cost_history = []
w_history = []
b_history = []

# Gradient Descent Loop
for i in range(trials):
    # 1. Prediction (y = wx + b)
    y_pred = x_train * w + b
    
    # 2. Cost Function (Mean Squared Error)
    cost = np.mean((y_train - y_pred) ** 2)
    
    # 3. Gradients nikalna
    dw = (-2 / num_samples) * np.sum(x_train * (y_train - y_pred))
    db = (-2 / num_samples) * np.sum(y_train - y_pred)
    
    # 4. Weights update karna
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 
    cost_history.append(cost)
    w_history.append(w)
    b_history.append(b)

print(f"Final Weights -> w: {w:.4f}, b: {b:.4f}")

# Final Fitted Line Plot
plt.scatter(x_train, y_train, color="blue")
plt.plot(x_train, x_train * w + b, color="red", label="Fitted Line")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(trials), cost_history, color='purple', linewidth=2)
plt.title("Gradient Descent: Cost vs Trials")
plt.xlabel("Iterations / Trials")
plt.ylabel("Cost (MSE)")
plt.grid(True)
plt.show()