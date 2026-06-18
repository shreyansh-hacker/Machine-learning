"""
================================================================================
Module Name: plr-root_cube_ft.py
Description: Regression using custom engineered non-linear features.
Methodology:
  1. Read input data from CSV.
  2. Engineer new feature columns: x, sqrt(x), x^3, and exp(x).
  3. Standardize inputs to prevent optimization problems.
  4. Optimize weights using gradient descent and plot fitting line.
================================================================================
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

filepath = r"C:/Users/shrey/Desktop/intenship_project/polynomial_data.csv"
data =pd.read_csv(filepath)
x = data ["x_train"].values
y = data["y_train"].values
alpha = 0.01

x_train=x.reshape(-1,1)
y =y.reshape(-1,1)
m= x.shape[0]
cost_history=[]
w=np.array([[1]])
b=np.array([[0]])
# print(w.shape,x_train.shape) 

d=4
x=np.zeros((m,d))
x[:,0] =x_train[:,0]
x[:,1] =x_train[:,0]**(1/2)
x[:,2] =x_train[:,0]**3
x[:,3] = np.exp(x_train[:,0])


mean =np.mean(x,axis=0)
st = np.std(x,axis=0)
x=(x-mean)/st


w= np.random.rand(1,d)

for _ in range(10000):
    y_ = (w@x.T+b).T
    cost = np.sum((y_-y)**2)/(2*m)
    cost_history.append(cost)
    djdw= ((x.T@(y_-y))/m).T
    djdb=np.mean(y_-y)
    w=w-alpha*djdw
    b=b-alpha*djdb
y_ = (w@x.T +b ).T

plt.scatter(x_train,y)
plt.plot(x_train,y_)
plt.show()
plt.plot(cost_history)
plt.show()