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

m =num_samples
sum_x=np.sum(x_train)
sum_y=np.sum(y_train)
sum_xx = np.sum(x_train **2)
sum_xy= np.sum(x_train*y_train)

w_formula = (m*sum_xy-sum_x*sum_y)/(m*sum_xx-(sum_x) **2)

b_formula = (sum_y-w_formula*sum_x)/m

print("w = ", w_formula)
print("b=",b_formula)