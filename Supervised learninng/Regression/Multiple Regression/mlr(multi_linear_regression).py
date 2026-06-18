"""
================================================================================
Module Name: mlr(multi_linear_regression).py
Description: Multiple Linear Regression from scratch.
Methodology:
  1. Generate synthetic 3-feature dataset.
  2. Predict output using vectorized matrix multiplication (X @ W + b).
  3. Compute MSE cost and parameter gradients.
  4. Update weights and plot cost minimization.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 100
num_features = 3

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, num_features)

# Define true weights (slopes) and intercept
true_weights = np.array([[3], [2], [1]])  # shape: (3, 1)
true_intercept = 4

# Generate noise
noise = np.random.randn(num_samples, 1)

# Compute y values
y_train = true_intercept + x_train @ true_weights + noise  # @ is matrix multiplication

# Plotting using only first feature for visualization
plt.figure(figsize = (8, 5))
plt.scatter(x_train[:, 0], y_train, color = "green", label = "Training data (vs 1st feature)", alpha = 0.7)
plt.title("Dummy Linear Regression Data (3 features)")
plt.xlabel("x_train[:, 0] (1st feature)")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)

W =np.zeros((num_features ,1))
b =0
learning_rate = 0.01
trials=1000
cost_history=[]
for i in range (trials):
    y_pred = x_train @ W+b

    #cf..
    cost=np.mean((y_pred-y_train)**2)/2
    
    error = y_pred-y_train
    
    dw= (1/num_samples)*(x_train.T@error)
    db = (1/num_samples)*np.sum(error)
    
    W= W-learning_rate*dw
    b=b-learning_rate*db
    cost_history.append(cost)

print("w =",W)
print("b=",b)
print("cost", cost)

f= x_train @ W + b
print("function",f)

plt.figure(figsize=(8,4))
plt.plot(range(trials),cost_history,color = 'purple' ,linewidth=2)
plt.xlabel("cost")
plt.ylabel("trials")
plt.grid(True)
plt.show()
