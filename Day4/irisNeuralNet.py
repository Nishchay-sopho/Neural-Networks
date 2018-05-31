import numpy as np
import pandas as pd

df=pd.read_csv('iris.csv')
X=df2.iloc[:,0:4].values
Y=df2.iloc[:,4].values
pd.get_dummies(Y,prefix=['Class'])

xPredicted=df2.iloc[0:2,0:2].values
X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)

class Neural_Network(object):
  def __init__(self):

    self.inputSize = 2
    self.outputSize = 3
    self.hiddenSize = 10


    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def forward(self, X):

    return o

  def sigmoid(self, s):

    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):

    return s * (1 - s)

  def backward(self, X, y, o):


  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print ("Input (scaled):" + str(xPredicted))
    print ("Output:" + str(self.forward(xPredicted)))
NN = Neural_Network()
for i in range(10000):
  NN.train(X, Y)

NN.predict()

