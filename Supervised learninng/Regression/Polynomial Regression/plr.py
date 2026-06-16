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

d=17
x=np.zeros((m,d))
for i in range(1,d+1):
    x[:,i-1] =x_train[:,0]**i

mean =np.mean(x,axis=0)
st = np.std(x,axis=0)
x=(x-mean)/st


w= np.random.rand(1,d)

for _ in range(1000):
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