import sys
from tasks import *
import numpy as np
# Script Usage: python3 test.py <seed>
# Read task number and seed value from command line

if __name__ == "__main__":
	seed=int(sys.argv[1])

	np.random.seed(int(seed))
	taskMnist()