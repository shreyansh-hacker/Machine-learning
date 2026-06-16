import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Generate dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=42
)

# Degree of polynomial
d = 5

# Create pipeline
model = Pipeline([
    ('poly', PolynomialFeatures(degree=d, include_bias=False)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y)

# Accuracy
print("Accuracy:", model.score(X, y))

# Create mesh grid
x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

grid = np.c_[xx1.ravel(), xx2.ravel()]

# Predict probabilities
pred = model.predict_proba(grid)[:,1]
pred = pred.reshape(xx1.shape)

# Plot decision region
plt.figure(figsize=(8,6))

plt.contourf(xx1, xx2, pred, levels=50, alpha=0.4)

plt.scatter(
    X[y==0][:,0],
    X[y==0][:,1],
    color="red",
    label="Class 0"
)

plt.scatter(
    X[y==1][:,0],
    X[y==1][:,1],
    color="blue",
    label="Class 1"
)

# Decision boundary
plt.contour(
    xx1, xx2,
    pred,
    levels=[0.5],
    colors='black'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Polynomial Logistic Regression (Scikit-Learn)")
plt.legend()
plt.grid(True)
plt.show()