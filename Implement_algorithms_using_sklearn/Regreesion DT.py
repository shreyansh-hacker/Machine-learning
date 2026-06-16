import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

# ================= DATA GENERATION =================

np.random.seed(42)

num_samples = 100

X = 2 * np.random.rand(num_samples, 1)

true_slope = 3
true_intercept = 4

noise = np.random.randn(num_samples, 1)

y = true_intercept + true_slope * X + noise

# ================= PLOT DATA =================

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data")
plt.grid(True)
plt.show()

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================= TRAIN DECISION TREE REGRESSOR =================

reg = DecisionTreeRegressor(
    max_depth=3,      # same as your depth = 3
    random_state=42
)

reg.fit(X_train, y_train)

# ================= PREDICTION =================

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

# ================= TREE VISUALIZATION =================

plt.figure(figsize=(14,8))

plot_tree(
    reg,
    feature_names=["x"],
    filled=True
)

plt.show()

# ================= REGRESSION CURVE =================

X_grid = np.linspace(
    X.min(),
    X.max(),
    500
).reshape(-1,1)

y_grid = reg.predict(X_grid)

plt.figure(figsize=(8,5))

plt.scatter(X, y, alpha=0.6, label="Data")

plt.plot(
    X_grid,
    y_grid,
    color="red",
    linewidth=3,
    label="Decision Tree Prediction"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Decision Tree Regression")
plt.legend()
plt.grid(True)

plt.show()