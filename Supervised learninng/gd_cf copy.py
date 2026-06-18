"""
================================================================================
Module Name: gd_cf copy.py
Description: Loop-based Simple Linear Regression with Gradient Descent.
Methodology:
  1. Generate synthetic 1D linear data.
  2. Loop explicitly over each data point to compute prediction error, accumulated cost (MSE), and gradients.
  3. Update slope and intercept weights iteratively.
  4. Plot fitted line and cost history.
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
m = x_train.shape[0]

cost_history = []
w_history = []
b_history = []

# Gradient Descent Loop
for i in range(trials):
    y_pred = []
    error = 0
    dw = 0
    db = 0
    # 1. Prediction (y = wx + b)
    for j in range(m):
        y_ = x_train[j][0] * w + b
        y_pred.append(y_)
    
        # 2. Cost Function (Mean Squared Error)
        error = error + ((y_train[j] - y_) ** 2)
    
        # 3. Gradients nikalna
        dw = dw + (x_train[j] * (y_ - y_train[j]))
        db = db + (y_ - y_train[j])

    cost = sum(error) / (2 * m)
    dw = sum(dw)/m
    db = sum(db)/m
    
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