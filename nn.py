import numpy as np
import random
from util import oneHotEncodeY
import itertools

class NeuralNetwork:

	def __init__(self, out_nodes, alpha, batchSize, epochs):
		# Method to initialize a Neural Network Object
		# Parameters
		# out_nodes - number of output nodes
		# alpha - learning rate
		# batchSize - Mini batch size
		# epochs - Number of epochs for training
		self.alpha = alpha
		self.batchSize = batchSize
		self.epochs = epochs
		self.layers = []
		self.out_nodes = out_nodes

	def addLayer(self, layer):
		# Method to add layers to the Neural Network
		self.layers.append(layer)

	def train(self, trainX, trainY, validX=None, validY=None, printTrainStats=True, printValStats=True, saveModel=False, loadModel=False, modelName=None):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training
		
		if loadModel:
			model = np.load(modelName)
			k,i = 0,0
			for l in self.layers:
				if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
					self.layers[i].weights = model[k]
					self.layers[i].biases = model[k+1]
					k+=2
				i+=1
			print("Model Loaded... ")

		for epoch in range(self.epochs):
			# A Training Epoch
			if printTrainStats or printValStats:
				print("Epoch: ", epoch)

			# Shuffle the training data for the current epoch
			X = np.asarray(trainX)
			Y = np.asarray(trainY)
			perm = np.arange(X.shape[0])
			np.random.shuffle(perm)
			X = X[perm]
			Y = Y[perm]

			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches
			numBatches = int(np.ceil(float(X.shape[0]) / self.batchSize))
			for batchNum in range(numBatches):
				# print(batchNum,"/",numBatches) # UNCOMMENT if model is taking too long to train
				XBatch = np.asarray(X[batchNum*self.batchSize: (batchNum+1)*self.batchSize])
				YBatch = np.asarray(Y[batchNum*self.batchSize: (batchNum+1)*self.batchSize])

				# Calculate the activations after the feedforward pass
				activations = self.feedforward(XBatch)	

				# Compute the loss	
				loss = self.computeLoss(YBatch, activations)
				trainLoss += loss
				
				# Estimate the one-hot encoded predicted labels after the feedword pass
				predLabels = oneHotEncodeY(np.argmax(activations[-1], axis=1), self.out_nodes)

				# Calculate the training accuracy for the current batch
				acc = self.computeAccuracy(YBatch, predLabels)
				trainAcc += acc
				# Backpropagation Pass to adjust weights and biases of the neural network
				self.backpropagate(activations, YBatch)

			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			if printTrainStats:
				print("Epoch ", epoch, " Training Loss=", loss, " Training Accuracy=", trainAcc)
			
			if saveModel:
				model = []
				for l in self.layers:
					print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				np.save(modelName, model)
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if validX is not None and validY is not None and printValStats:
				_, validAcc = self.validate(validX, validY)
				print("Validation Set Accuracy: ", validAcc, "%")

	def computeLoss(self, Y, predictions):
		# Returns the crossentropy loss function given the prediction and the true labels Y

		# print(predictions)
		final_pred = predictions[-1]
		# final_pred = np.asarray(final_pred)
		# print("final_pred", final_pred.shape)
		# print("Y", type(Y))
		# print("Y", Y.shape)
		# print(final_pred)
		# print(Y)
		loss = np.sum(-1.0*np.multiply(Y, np.log(final_pred)))
		return loss
		# raise NotImplementedError

	def computeAccuracy(self, Y, predLabels):
		# Returns the accuracy given the true labels Y and predicted labels predLabels
		correct = 0
		for i in range(len(Y)):
			if np.array_equal(Y[i], predLabels[i]):
				correct += 1
		accuracy = (float(correct) / len(Y)) * 100
		return accuracy

	def validate(self, validX, validY):
		# Input 
		# validX : Validation Input Data
		# validY : Validation Labels
		# Returns the validation accuracy evaluated over the current neural network model
		valActivations = self.feedforward(validX)
		pred = np.argmax(valActivations[-1], axis=1)
		validPred = oneHotEncodeY(pred, self.out_nodes)
		validAcc = self.computeAccuracy(validY, validPred)
		return pred, validAcc

	def feedforward(self, X):
		# Input
		# X : Current Batch of Input Data as an nparray
		# Output
		# Returns the activations at each layer(starting from the first layer(input layer)) to 
		# the output layer of the network as a list of np multi-dimensional arrays
		# Note: Activations at the first layer(input layer) is X itself		
		
		# raise NotImplementedError

		ans = [X]
		data = X
		for layer in self.layers:
			data = layer.forwardpass(data)
			ans.append(data)

		# print("feedforward", ans)
		# ans1 = nps.array([nps.array(xi) for xi in ans])
		return ans

	def backpropagate(self, activations, Y):
		# Input
		# activations : The activations at each layer(starting from second layer(first hidden layer)) of the
		# neural network calulated in the feedforward pass
		# Y : True labels of the training data
		# This method adjusts the weights(self.layers's weights) and biases(self.layers's biases) as calculated from the
		# backpropagation algorithm
		# Hint: Start with derivative of cross entropy from the last layer

		# raise NotImplementedError
		numLayers = len(self.layers)
		# delta = []
		# for i in self.layers:
		# 	delta.append(np.random.normal(0, 1, (self.batchSize, i.in_nodes)))

		# delta[numLayers-1] = activations[-1]*(1 - activations[-1]) * (Y - activations[-1])

		# softmax_Y = np.log(1+ np.exp(Y))
		# delta[numLayers-1] = -np.divide(Y, softmax_Y) - np.divide(1 - Y, 1- softmax_Y)
		delta = -np.divide(Y,activations[-1])
		# delta[numLayers -1] = -1*np.divide(Y,activations[-1])

		# for k in range(numLayers-2, -1, -1):
		# 	delta[k] = activations[k]*(1-activations[k])* np.dot(delta[k+1], self.layers[k].weights.T)
		# 	self.layers[k].weights += self.alpha * (np.dot(activations[k].T, delta[k+1]) / self.batchSize)
		# 	self.layers[k].biases += self.alpha * delta[k+1].mean(axis=0)
		for k in range(numLayers-1, -1, -1):
			delta = self.layers[k].backwardpass(self.alpha, activations[k], delta)