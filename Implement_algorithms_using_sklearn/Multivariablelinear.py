import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set seed
np.random.seed(42)

# Number of samples and features
num_samples = 100
num_features = 3

# Generate random features
X = 2 * np.random.rand(num_samples, num_features)

# True weights and intercept
true_weights = np.array([[3], [2], [1]])
true_intercept = 4

# Noise
noise = np.random.randn(num_samples, 1)

# Target variable
y = true_intercept + X @ true_weights + noise
y = y.ravel()  # Convert (100,1) -> (100,)

# Plot using first feature
plt.figure(figsize=(8,5))
plt.scatter(
    X[:,0],
    y,
    color="green",
    alpha=0.7,
    label="Training Data"
)

plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("Multiple Linear Regression Data")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Learned parameters
print("Intercept:", model.intercept_)
print("Weights:", model.coef_)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R² Score:", r2)

# Plot actual vs predicted
plt.figure(figsize=(8,5))
plt.scatter(y, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()