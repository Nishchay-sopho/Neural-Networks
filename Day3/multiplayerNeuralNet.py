import numpy as np

X = np.array(([0,1], [1,1],[0,0],[1,0]), dtype=float)
y = np.array(([1], [0],[0],[1]), dtype=float)
xPredicted = np.array(([1,0],[0,1],[1,1]), dtype=float)

class Neural_Network(object):
	def __init__(self):
		self.inputSize=2
		self.numberOfHiddenLayers=2
		self.outputSize=1
		self.hiddenSizes=np.array([3,3])

		self.weights = list([])
		self.weights.append (np.random.randn(self.inputSize,self.hiddenSizes[0]))
		for i in range(0,self.numberOfHiddenLayers-1) :
			self.weights.append(np.random.randn(self.hiddenSizes[i], self.hiddenSizes[i+1]))
		print (self.weights)

NN= Neural_Network()
