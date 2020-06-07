import numpy as np
import gzip
import _pickle as cPickle

def oneHotEncodeY(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
	return (np.eye(nb_classes)[Y]).astype(int)

def readMNIST():
	# f = gzip.open('datasets/mnist.pkl.gz', 'rb')
	# train_set, val_set, test_set = cPickle.load(f, encoding='latin1')
	# f.close()
	# trainX, trainY = train_set

	trainX, trainY = myfunc()
	# print(type(trainX))
	# print("trainX shape", trainX.shape)
	# print("trainY shape", trainY.shape)
	# print("trainX index 0", trainX[1,:])
	# zi = 5
	# y = np.reshape(trainX[zi,:], (784,1))
	# y = np.reshape(y, (28,28))
	# y = 1*(y >= 0.5)
	# print(y.shape)
	# for i in range(784):
	# print(y)
	# for i in range(28):
	# 	for j in range(28):
	# 		if (y[i][j] == 1): print("X", end = " ")
	# 		else: print("", end = " ")
	# 	print("\n")
	# print("trainY val", trainY[zi])

	# valX, valY = val_set
	# testX, testY = test_set
	
	valX = trainX
	valY = trainY

	
	testX = trainX[1400:1493]
	testY = trainY[1400:1493]
	 
	trainX = np.where(trainX>0, 1, 0)
	valX = np.where(valX>0, 1, 0)
	testX = np.where(testX>0, 1, 0)

	XTrain = np.array(trainX)
	YTrain = np.array(oneHotEncodeY(trainY, 36))
	XVal = np.array(valX)
	YVal = np.array(oneHotEncodeY(valY, 36))
	XTest = np.array(testX)
	YTest = np.array(oneHotEncodeY(testY, 36))

	return XTrain[0:1400], YTrain[0:1400], XVal[500:1200], YVal[500:1200], XTest, YTest

def split(X,Y):
	Y=Y.astype(int)
	perm = np.arange(X.shape[0])
	np.random.shuffle(perm)
	X = X[perm]
	Y = Y[perm]
	
	trainX = X[0:8000]
	trainY = Y[0:8000]
	XTrain = np.array(trainX)
	YTrain = np.array(oneHotEncodeY(trainY,2))
	
	valX = X[8000:9000]
	valY = Y[8000:9000]
	XVal = np.array(valX)
	YVal = np.array(oneHotEncodeY(valY,2))

	testX = X[9000:10000]
	testY = Y[9000:10000]
	XTest = np.array(testX)
	YTest = np.array(oneHotEncodeY(testY,2))

	return XTrain, YTrain, XVal, YVal, XTest, YTest

def extractGrid(filename):
	x = np.zeros((20,20))
	i = 0

	with open(filename) as file_handler:	
		for line in file_handler:
			w = line.split(" ")
			for wx in range(20):
				x[i][wx] = float(w[wx])
			i+=1

	# print(x)
	# display(x)
	x = np.reshape(x, (400,1))
	return list(x)

def display(x):
	x = 1*(x > 0.5)
	for i in range(20):
		for j in range(20):
			if x[i][j] == 1:print("X"),
			else:print(" "), 
		print("\n")

def nod(i):
	x = ord(i)
	if x < 60:
		return x - 48
	else:
		return x - 65 + 10

def myfunc():
	dirname = "trained/"
	filename = "trained/data.txt"

	characters = []

	with open(filename) as f:
		for l in f:
			characters.append(l[0])

	trainX = []
	trainY = []

	for i in range(len(characters)):
		if characters[i] != '-':
			trainX.append(extractGrid(dirname + str(i+1) + ".txt"))
			
			trainY.append(nod(str(characters[i])))

	trainX = np.array(trainX)
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
	trainY = np.array(trainY)

	return trainX, trainY