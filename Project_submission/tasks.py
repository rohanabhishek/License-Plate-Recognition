import nn
import numpy as np
import sys

from util import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	# raise NotImplementedError
	modelName = 'my_model.npy'
	# print(XTrain.shape)
	# print(YTrain.shape)
	nn1 = nn.NeuralNetwork(36, 0.001, 200, 10)
	nn1.addLayer(FullyConnectedLayer(400, 50, "relu"))
	nn1.addLayer(FullyConnectedLayer(50, 36, "softmax"))
	###############################################
	# return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION
	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=True, saveModel=True, modelName=modelName)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest