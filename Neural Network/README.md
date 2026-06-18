# Neural Networks from Scratch & Frameworks

This directory focuses on building, training, and optimizing Artificial Neural Networks, specifically **Multi-Layer Perceptrons (MLPs)**.

## File Registry & Audit

*   **`fmnist.py`**: Manually implemented MLP from scratch (using NumPy) to classify the **Fashion-MNIST** dataset. Features activation functions (ReLU, Sigmoid, Softmax) and manual backpropagation gradient computations.
*   **`mnist.py`**: Manual MLP from scratch on the **MNIST** dataset. Implements dataset splitting, one-hot encoding, normalization, and full forward/backward propagation.
*   **`practice.py`**: MLP containing custom PIL-based data augmentation (image rotation and shifting) inside the training loop to study model robustness.
*   **`norm_nn.py`**: Analysis script demonstrating feature scaling requirements on the coffee roasting dataset.
*   **`mnist.ipynb` / `norm_nn.ipynb` / `pca_mnist.ipynb` / `practice_nn.ipynb`**: Interactive notebooks demonstrating training, normalization, and dimensional reduction (PCA) on digit datasets.

## Datasets Included
*   `fashion-mnist_train.csv` / `fashion-mnist_test.csv`
*   `mnist_train.csv` / `mnist_test.csv`
