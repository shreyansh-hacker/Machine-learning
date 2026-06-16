import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Set seed
np.random.seed(42)

# Generate data
num_samples = 100

x_train = 2 * np.random.rand(num_samples, 1)

true_slope = 3
true_intercept = 4

noise = np.random.randn(num_samples, 1)

y_train = true_intercept + true_slope * x_train + noise

# ===========================
# KNN Regression
# ===========================

k = 5

model = KNeighborsRegressor(
    n_neighbors=k+1  # +1 because point itself is included
)

model.fit(x_train, y_train.ravel())

# Predict on training data
y_pred = model.predict(x_train)

# ===========================
# Plot
# ===========================

plt.figure(figsize=(8,5))

plt.scatter(
    x_train,
    y_train,
    color="blue",
    label="Training Data",
    alpha=0.7
)

plt.scatter(
    x_train,
    y_pred,
    color="red",
    label="Predictions",
    alpha=0.7
)

plt.title("KNN Regression")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)

plt.show()