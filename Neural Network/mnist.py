"""
================================================================================
Module Name: mnist.py
Description: Multi-Layer Perceptron (MLP) from scratch for digit classification (MNIST).
Methodology:
  1. Load dataset, perform stratified splitting manually per digit label.
  2. Standardize input pixels using mean and standard deviation.
  3. Implement forward and backward propagation manually.
  4. Track training/testing cost and accuracy over epochs.
  5. Plot optimization progression curves.
================================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def softmax(z):
    z = z - np.max(z,axis=0,keepdims=True)
    return np.exp(z)/(np.sum(np.exp(z),axis=0,keepdims=True) + 1e-10)
def relu(z):
    return np.maximum(z,0)
def r_derivative(z):
    return np.where(z>0,1,0)
def sigmoid(z_):
        return 1/(1+np.exp(-z_))
def der_sigmoid(z):
        return sigmoid(z)*(1-sigmoid(z))
# Load the fixed CSV
df = pd.read_csv("mnist_train.csv")
df1 = pd.read_csv("mnist_test.csv")
# Group by class label
grouped = df.groupby("label")

# Prepare empty lists for train and test splits
train_list = []
test_list = []

# Split each class group into 80% train and 20% test
for label, group in grouped:
    train_split, test_split = train_test_split(
        group, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True,
        stratify=None  # We are already splitting by class manually
    )
    train_list.append(train_split)
    test_list.append(test_split)

# Concatenate all class-wise splits
train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)


# One example per class
examples = train_df.groupby("label").first().reset_index()

# # Plot
# plt.figure(figsize=(10, 4))
# for i in range(10):
#     ax = plt.subplot(2, 5, i + 1)
#     img = examples.loc[i].drop("label").values.astype(np.uint8).reshape(28, 28)
#     plt.imshow(img, cmap="gray")
#     plt.title(f"Label: {examples.loc[i, 'label']}")
#     plt.axis("off")

# plt.tight_layout()
# # plt.show()

y = df["label"].values#.reshape(-1,1)
#print(y.shape)
X= df.iloc[0:,1:].values
#print(X.shape)
#train
y_ = np.eye(10)[y] # y one hot
X = (X-X.mean()) / X.std()


#test

X_test=df1.iloc[0:,1:].values
y_test1=df1["label"]
y_test=np.eye(10)[y_test1]
X_test=(X_test-X.mean())/X.std()


layers = [128,64,10]
learning_rate = 0.01
W=[]
b=[]
inputs=X.shape[1]
activations=["relu","relu","softmax"]

m = X.shape[0]
# W values 
for L in range(len(layers)):
    if L>0:
        inputs=layers[L-1]
    w = np.random.randn(layers[L], inputs) * np.sqrt(2/inputs)
    W.append(w)
    b_=np.zeros((layers[L],1))
    b.append(b_)
cost_history = []
cost_history_test=[]
acc_history = []
acc_history_test=[]
#iterations
for epoch in range(500):
    Z=[]
    A_=X.T
    A=[]
    
    #forwardprop train data
    for i in range(len(layers)):
        z = W[i]@A_+b[i]
        if i==len(layers)-1:
            A_ = softmax(z)
        else:
            if activations[i]=="relu":
               A_ = relu(z)
            if activations[i]=="sigmoid":
                A_ = sigmoid(z)   
        Z.append(z)
        A.append(A_)

    cost = -np.sum(y_ * np.log(A[-1].T + 1e-10)) / m
    #print("Training_cost",cost)
    cost_history.append(cost)

    y_pred=np.argmax(A_.T,axis=1)
    acc = np.mean(y_pred == y)*100
    acc_history.append(acc)
    #print("Accuracy for training",acc)
    # backward prop       

    Dw=[]
    Db=[]
 
    for i in range (len(layers)-1,-1,-1):
        if (i == (len(layers)-1)):
            delta = A[i] - y_.T
        else:
            if activations[i] == "sigmoid":
                delta = W[i+1].T@delta*(der_sigmoid(Z[i]))
            if activations[i] == "relu":
                delta = W[i+1].T@delta*(r_derivative(Z[i]))
        if i!=0:
            dw = (delta@(A[i-1]).T)/m
        else:
            dw = (delta@X)/m
        Dw.append(dw)
        db = np.sum(delta, axis=1, keepdims=True) / m
        Db.append(db)
    Dw=Dw[::-1]
    Db=Db[::-1]
    for j in range(len(layers)):         
         W[j] = W[j]- learning_rate*Dw[j]
         b[j] = b[j]- learning_rate*Db[j]
    
    A_=X_test.T
    #forward-> test data
    for i in range (len(layers)):
        z = W[i]@A_+b[i]
        if i==len(layers)-1:
            A_ = softmax(z)
        else:
            if activations[i]=="relu":
               A_ = relu(z)
            if activations[i]=="sigmoid":
                A_ = sigmoid(z)   

    cost = -np.sum(y_test * np.log(A_.T + 1e-10)) / m
    #print("Testing_cost",cost)
    cost_history_test.append(cost)

    #backward prop -> test 
    y_pred=np.argmax(A_.T,axis=1)
    acc = np.mean(y_pred == y_test1)*100
    acc_history_test.append(acc)
    #print("Accuray for testing",acc)
        
    print("train_cost:",cost_history[-1],"train_acc:",acc_history[-1],"test_cost:",cost_history_test[-1],"test_acc:",acc_history_test[-1])

plt.plot(cost_history,label="train_cost")
plt.plot(cost_history_test,label="train_acc")
plt.show()
plt.plot(acc_history,label="test_cost")
plt.plot(acc_history_test,label="test_acc")
plt.show()
