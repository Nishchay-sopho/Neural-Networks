import numpy as np
from keras.datasets import mnist
import pickle as pkl
data = mnist.load_data()

(trainX,trainY),(testX,testY) = data

trainX = trainX.reshape((60000,784))/255 - 0.5
testX = testX.reshape((10000,784))/255 - 0.5

def onehot(T):
    train_y = np.zeros((T.shape[0],10))
    for i,d in enumerate(T):
        train_y[i,d-1] = 1
    return train_y

trainY = onehot(trainY)
testY = onehot(testY)
    
def _gen_layer_weights(m, n):
    return np.random.normal(0,1,(m,n))

class MultilayerNeuralNet:
    def __init__(self, input_size, output_layer, hidden_layers):
        self.w = []
        layers = len(hidden_layers)
        if layers == 1:
            self.w.append(_gen_layer_weights(input_size+1,hidden_layers[0]))
            self.w.append(_gen_layer_weights(hidden_layers[-1]+1,output_layer))
        else:
            self.w.append(_gen_layer_weights(input_size+1,hidden_layers[0]))
            for i in range(1,layers):
                self.w.append(_gen_layer_weights(hidden_layers[i-1]+1,hidden_layers[i]))
            self.w.append(_gen_layer_weights(hidden_layers[-1]+1,output_layer))
        self.nlayers = len(self.w)
        
def _sigmoid(self,X):
    return (1/(1+np.exp(-X)))

MultilayerNeuralNet._sigmoid = _sigmoid

def _sigmoid_prime(self, X):
    return self._sigmoid(X)*self._sigmoid(1-X)

MultilayerNeuralNet._sigmoid_prime = _sigmoid_prime
                
def train(self,X,Y,LR,epochs):
    j = np.zeros((epochs,1))
    m = Y.shape[0]
    for i in range(epochs):
        out = self._train(X,Y,LR)
        acc = np.sum(np.argmax(out,axis=1) == np.argmax(Y,axis=1))/m
        print("%d - Acc: %f"%(i,acc))
    return j

MultilayerNeuralNet.train = train
        

def _cost(self, out, Y):
    return np.sum(-Y*np.log(out) - (1-Y)*np.log(1-out))

MultilayerNeuralNet._cost = _cost

def _train(self,X,Y,LR):
    m = Y.shape[0]
    layers = [None]*self.nlayers
    lin = X
    for i,t in enumerate(self.w):
        lin2 = np.column_stack((np.ones((lin.shape[0],1)),lin))
        out = self._sigmoid(lin2@t)
        layers[i] = (lin,out,lin2)
        lin = out
    
    grad = [None]*self.nlayers
    
    delta_o = (Y - layers[-1][1])*self._sigmoid_prime(layers[-1][1])
    grad[-1] = (1/m)*(layers[-1][2].transpose()@delta_o)

    for i in range(2,len(layers)+1):
        delta_o = (delta_o@self.w[1-i].transpose())[:,1:]*self._sigmoid_prime(layers[-i][1])
        grad[-i] = (1/m)*(layers[-i][2].transpose()@delta_o)
        
    for i in range(len(layers)):
        self.w[i] += LR*grad[i]
    
    return out
        
MultilayerNeuralNet._train = _train

def predict(self,X):
    lin = X
    for t in self.w:
        lin2 = np.column_stack((np.ones((lin.shape[0],1)),lin))
        lin = self._sigmoid(lin2@t)
    return lin

MultilayerNeuralNet.predict = predict    

n = MultilayerNeuralNet(784, 10, (256,32))

j = n.train(trainX, trainY, 0.1, 10000)
print (str(predict(testX)))
