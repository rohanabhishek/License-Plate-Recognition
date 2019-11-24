import sys
import numpy as np

fn = str(sys.argv[1]) + ".txt"

x = np.zeros((20,20))
i = 0

with open(fn) as f:	
	for line in f:
		w = line.split(" ")
		for wx in range(20):
			x[i][wx] = float(w[wx])
		i+=1	

x = 1*(x > 0.5)

for i in range(20):
	for j in range(20):
		if x[i][j] == 1:print("X"),
		else:print(" "), 
	print("\n")
