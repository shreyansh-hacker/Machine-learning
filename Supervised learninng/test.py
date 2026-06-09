import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


df = pd.read_csv("fashion-mnist_train.csv")

X = df.iloc[:,1:].values
y = df.iloc[0:,0].values

#80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train =X_train/255
X_test = X_test/255
y_train= np.eye(10)[y_train]
y_test= np.eye(10)[y_test]
def softmax (z):
    z=z-np.max(z,axis=0,keepdims=True)
    return np.exp(z)/np.sum(np.exp(z),axis=0,keepdims=True)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def der_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
def relu(z):
    return np.maximum(0,z)
def der_relu(z):
    return np.where(z>0,1,0)
W=[]
b=[]
activations = ["relu", "softmax"]
layers = [128, 10]
inputs=X.shape[1]
m_test = X_test.shape[0]
m_train=X_train.shape[0]
#W value
for L in range(len(layers)):
    if L>0:
      inputs=layers[L-1]
    w=(np.random.randn(layers[L],inputs)*np.sqrt(2/inputs))
    W.append(w)
    b_= np.zeros((layers[L],1))
    b.append(b_)

cost_history=[]
cost_history_test=[]
acc_history=[]
acc_history_test=[]

for epoch in range(500):
  # image_data = X_train.values.astype('uint8').reshape(28, 28)
  df = pd.read_csv('fashion-mnist_train.csv')
  image_data = df.iloc[0, 1:].values.astype('uint8').reshape(28, 28)
  print(image_data.shape)
  img = Image.fromarray(image_data)

  img_rotate = img.rotate(15)

  img_shift = ImageChops.offset(img_rotate, 3, 0)

  Aug_data = img_shift.reshape(-1,784)

  A_=Aug_data.T
  A=[]
  Z=[]
  #forward
  learning_rate=0.01
  for ii in range(len(layers)):
    z=W[ii]@A_+b[ii]
    if ii==len(layers)-1:
       A_=softmax(z)
    else:
      if activations[ii]=="relu":
         A_=relu(z)
      elif activations[ii]=="sigmoid":
         A_=sigmoid(z)
    Z.append(z)
    A.append(A_)
  cost = -np.sum(y_train*np.log(A[-1].T))/m_train
  cost_history.append(cost)
  y_pred=np.argmax(A[-1].T,axis=1)
  acc=np.mean(y_pred==np.argmax(y_train,axis=1))*100
  acc_history.append(acc)
  #backward
  Dw=[]
  Db=[]
  for i in range (len(layers)-1,-1,-1):
    if i==len(layers)-1:
      delta=A[-1]-y_train.T
    else:
      if activations[i]=="relu":
         delta=W[i+1].T@delta*der_relu(Z[i])
      elif activations[i]=="sigmoid":
         delta=W[i+1].T@delta*der_sigmoid(Z[i])
    if i!=0:
        dw=delta@A[i-1].T/m_train
    else:
        print(delta.shape,X_train.shape)
        dw=(delta@X_train)/m_train
    Dw.append(dw)
    db=np.sum(delta,axis=1,keepdims=True)/m_train
    Db.append(db)
  Dw=Dw[::-1]
  Db=Db[::-1]
  for j in range(len(layers)):
    W[j]=W[j]-learning_rate*Dw[j]
    b[j]=b[j]-learning_rate*Db[j]
    
  #test
  A_=X_test.T
  A = [] 
  Z = []
  for ii in range(len(layers)):
    z=W[ii]@A_+b[ii]
    if ii==len(layers)-1:
       A_=softmax(z)
    else:
      if activations[ii]=="relu":
         A_=relu(z)
      elif activations[ii]=="sigmoid":
         A_=sigmoid(z)
    Z.append(z)
    A.append(A_)
  cost = -np.sum(y_test*np.log(A[-1].T))/m_test
  cost_history_test.append(cost)
  y_pred=np.argmax(A[-1].T,axis=1)
  acc=np.mean(y_pred==np.argmax(y_test,axis=1))*100
  acc_history_test.append(acc)
  print("train_cost",cost_history[-1],"train_accuracy",acc_history[-1],"test_cost",cost_history_test[-1],"test_acc_",acc_history_test[-1])

plt.plot(cost_history,label="train_cost")
plt.plot(cost_history_test,label="test_cost")
plt.legend()
plt.show() 
plt.plot(acc_history,label="train_acc")
plt.plot(acc_history_test,label="test_acc")
plt.legend()
plt.show()
