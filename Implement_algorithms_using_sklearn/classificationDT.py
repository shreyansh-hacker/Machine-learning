"""
================================================================================
Module Name: classificationDT.py
Description: Classification using Scikit-Learn's DecisionTreeClassifier.
Methodology:
  1. Generate synthetic binary classification dataset.
  2. Train a DecisionTreeClassifier using entropy as splitting criterion.
  3. Visualize tree nodes.
  4. Plot decision boundary region mesh showing classification partition.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ================= DATA GENERATION =================

np.random.seed(42)

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

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# ================= TRAIN DECISION TREE =================

clf = DecisionTreeClassifier(
    criterion='entropy',   # Information Gain
    random_state=42
)

clf.fit(X_train, y_train)

# ================= PREDICTION =================

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("Accuracy:", accuracy)

# ================= TREE VISUALIZATION =================

plt.figure(figsize=(14,8))

plot_tree(
    clf,
    filled=True,
    feature_names=["Feature 1", "Feature 2"],
    class_names=["0", "1"]
)

plt.show()

# ================= DECISION BOUNDARY =================

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

Z = clf.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.4,
    cmap="coolwarm"
)

plt.scatter(
    X[y==0][:,0],
    X[y==0][:,1],
    label="Class 0"
)

plt.scatter(
    X[y==1][:,0],
    X[y==1][:,1],
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.legend()
plt.grid(True)

plt.show()