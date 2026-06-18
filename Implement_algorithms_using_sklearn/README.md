# Scikit-Learn Machine Learning Implementations

This folder contains implementations of core machine learning algorithms utilizing the industry-standard **Scikit-Learn (sklearn)** library.

## File Registry & Audit

| File Name | ML Concept | Algorithm / Model | Description |
| :--- | :--- | :--- | :--- |
| **`AnomalyDetection.py`** | Anomaly Detection | `EllipticEnvelope` | Fits a robust covariance estimate to detect outliers on a synthetic coffee roasting dataset. |
| **`DecisionTree.py`** | Classification | `DecisionTreeClassifier` | Encodes categorical variables and trains a classification tree using Information Gain (Entropy). |
| **`Regreesion DT.py`** | Regression | `DecisionTreeRegressor` | Predicts continuous targets using a regression tree of depth 3. |
| **`classificationDT.py`** | Classification | `DecisionTreeClassifier` | Generates a 2D classification dataset, fits a tree, and visualizes the decision boundary. |
| **`KNN_Linear Regression.py`** | Regression | `KNeighborsRegressor` | Implements K-Nearest Neighbors regression on noisy linear data. |
| **`KNN_LogisticRegression.py`** | Classification | `KNeighborsClassifier` | Trains a KNN classifier and plots predictions on synthetic 2D binary data. |
| **`K_means_clustering.py`** | Clustering | `KMeans` | Performs centroid-based unsupervised clustering on classification features. |
| **`logisticregression.py`** | Classification | `LogisticRegression` | Performs binary classification on scaled features with analytical decision boundary plotting. |
| **`Multivariablelinear.py`** | Regression | `LinearRegression` | Fits a multiple linear regression model with three features. |
| **`polynomial_data.py`** | Regression | `Pipeline` | Fits high-degree (17) polynomial regression using scaling and linear regression pipeline. |
| **`polynomialregression.py`** | Classification | `Pipeline` | Builds a pipeline of `PolynomialFeatures` and `LogisticRegression` to model non-linear boundaries. |
| **`svmhinghloss.py`** | Classification | `SVC` (Linear Kernel) | Fits a linear SVM using high-degree polynomial features. |
| **`svmlagrangebased.py`** | Classification | `SVC` (RBF Kernel) | Solves non-linear classification using RBF kernel SVM, plotting support vectors. |
| **`svrhingloss.py`** | Regression | `Pipeline` | Polynomial regression using Linear Regression with degree-8 features. |
| **`svrlagrangebased.py`** | Regression | `SVR` (RBF Kernel) | Uses Support Vector Regression with RBF kernel to model non-linear relations. |

## Dataset Files
*   `animal_features (1).csv`: Target animal data for decision tree classification practice.
*   `polynomial_data (1).csv`: Dataset used for evaluating polynomial regression and support vector regression.
