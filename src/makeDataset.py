import sys
import numpy as np

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


dirname = sys.argv[1]
filename = sys.argv[2]

characters = []

with open(filename) as f:
	for l in f:
		characters.append(l[0])

trainX = []
trainY = []

for i in range(len(characters)):
	if characters[i] != '-':
		trainX.append(extractGrid(dirname + str(i+1) + ".txt"))
		trainY.append(ord(str(characters[i])))

trainX = np.array(trainX)
trainY = np.array(trainY)