import numpy as np
import pandas as pd

df=pd.read_csv('iris.csv')
X=df.iloc[:,0:4].values
df=pd.get_dummies(df,prefix=['Class'])
y=df.iloc[:,4:7].values
X = X/np.amax(X, axis=0)
#xPredicted = xPredicted/np.amax(xPredicted, axis=0)
#y = y/100

class Neural_Network(object):
  def __init__(self):

    self.inputSize = 4
    self.outputSize = 3
    self.hiddenSize = 10

    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
	self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def predict(self):
    l1 = 1/(1 + np.exp(-(np.dot(X, NN.W1))))
    l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
    print (np.round(l2,3))

NN = Neural_Network()

learning_rate = 0.2 # slowly update the network
for epoch in range(10000):
    l1 = 1/(1 + np.exp(-(np.dot(X, NN.W1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(NN.W2.T) * (l1 * (1-l1))
    NN.W2 += l1.T.dot(l2_delta) * learning_rate
    NN.W1 += X.T.dot(l1_delta) * learning_rate
    #print ('Error:', er)
NN.predict()
