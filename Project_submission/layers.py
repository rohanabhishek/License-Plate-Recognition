import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.activation = activation
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# ans = np.zeros((n, self.out_nodes), dtype = float)
		# ans = np.ndarray((n, self.out_nodes), dtype = float)
		# for i in range(n):
			# print("self.weights.T", self.weights.T.shape)
			# print("X", X.shape)
			# print("self.biases", self.biases.shape)
			# ans[i] = self.weights.T @ X[i] + self.biases

		ans = X@self.weights + self.biases
		if self.activation == 'relu':
			# print("X.shape", X.shape)
			# print(X)
			# raise NotImplementedError
			self.data = ans
			ans = relu_of_X(ans)
			return ans
		elif self.activation == 'softmax':
			# raise NotImplementedError
			# ans = X@self.weights + self.biases
			ans = softmax_of_X(ans)
			# print("softmax_Ans", ans.shape)
			self.data = ans
			return ans
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()

		
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		ans = np.zeros((n, 1), dtype = float)
		if self.activation == 'relu':
			inp_delta = delta*gradient_relu_of_X(self.data, delta)
			# term1 =	(activation_prev.T @ inp_delta)
			# gradsum = np.sum(inp_delta, axis = 0, keepdims = True)
			# print("inp_delta Relu" , inp_delta.shape)
			# print("activation_prev.T Relu", activation_prev.T.shape)
			# print("self.weights Relu", self.weights.shape)
			# print("term1.T Relu", term1.T.shape)
			# print("gradsum Relu" , gradsum.shape)
			# print("self.biases Relu", self.biases.shape)
			# new_delta = inp_delta @ self.weights.T
			# self.weights -= lr*term1
			# self.biases -= lr*gradsum
			# return new_delta
		elif self.activation == 'softmax':
			inp_delta = gradient_softmax_of_X(self.data, delta)
			# new_delta = inp_delta @ self.weights.T
			# print("inp_delta", inp_delta.shape)
			# print("activation_prev.T", activation_prev.T.shape)
			# print("self.weights", self.weights.shape)
			# print("term1.T", term1.T.shape)
			# print("gradsum", gradsum.shape)
			# print("self.biases", self.biases.shape)
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		
		term1 = (activation_prev.T @ inp_delta)
		gradsum = np.sum(inp_delta, axis = 0, keepdims = True)
		new_delta = inp_delta @ self.weights.T
		self.weights -= lr*term1
		self.biases -= lr*gradsum
		return new_delta
		# return ans

		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride, activation):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride
		self.activation = activation
		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# print("ConvolutionLayer")
		# print(self.in_depth)
		# print(self.in_row, self.in_col)
		# print(self.filter_row)
		# print(self.stride)
		# print(self.activation)
		# print(self.out_depth)
		# print(self.out_row)
		# print(self.out_col)
		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		ans = np.zeros((n,self.out_depth, self.out_row, self.out_col))
		num_out_depth = self.out_depth
		num_out_row = self.out_row
		num_out_col = self.out_col

		for batch in range(n):
			for r in range(num_out_row):
				for c in range(num_out_col):
					patch_now = X[batch, :, self.stride*r:self.stride*r+self.filter_row, self.stride*c:self.stride*c+self.filter_col]
					ans[batch,:,r,c] = np.sum(np.sum(np.sum(patch_now*self.weights, 1), 1), 1) + self.biases
				
		if self.activation == 'relu':
			self.data = ans
			ans = relu_of_X(ans)
			return ans

		elif self.activation == 'softmax':
			ans = softmax_of_X(ans)
			self.data = ans
			return ans
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()

		
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		new_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))

		if self.activation == 'relu':
			inp_delta = gradient_relu_of_X(self.data, delta)
			delta_cub = (delta * inp_delta)
		elif self.activation == 'softmax':
			delta_cub = gradient_softmax_of_X(self.data, data)
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		
		for batch in range(n):
			for r in range(self.in_row):
				for c in range(self.in_col):
					r_min, r_max = indices(r, True)
					c_min, c_max = indices(c, False)
					for i in range(r_min, r_max+1):
						for j in range(c_min,c_max+1):
							new_delta[batch,:,r,c] += (delta_cub[batch,:,i,j] @ self.weights[:,:,r-self.stride*i,c-self.stride*j])

		for batch in range(n):
			for depth_out in range(self.out_depth):
				dWeight = np.zeros((self.in_depth,self.filter_row,self.filter_col))
				dBias = 0
				for i in range(self.out_row):
					for j in range(self.out_col):
						dBias += delta_cub[batch,depth_out,i,j]
						dWeight += delta_cub[batch,depth_out,i,j]*activation_prev[batch,:,self.stride*i:self.stride*i+self.filter_row,self.stride*j:self.stride*j+self.filter_col]

				self.weights[depth_out] = self.weights[depth_out] - lr*dWeight
				self.biases[depth_out] = self.biases[depth_out] - lr*dBias
		
		return new_delta

	def indices(in_ind, is_row):
			if is_row:
				max_index = min(int(in_ind/self.stride), self.out_row-1)
				rem = (in_ind - self.filter_row) % self.stride
				quo = (in_ind - self.filter_row) / self.stride
			else:
				max_index = min(int(in_ind/self.stride), self.out_col-1)
				rem = (in_ind - self.filter_col) % self.stride
				quo = (in_ind - self.filter_col) / self.stride

			if rem == 0:
				min_index = int(max(0, quo + 1))
			else:
				min_index = int(max(0, int(np.ceil(quo))))
			return min_index, max_index

		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		num_out_depth = self.out_depth
		num_out_row = self.out_row
		num_out_col = self.out_col
		ans = np.zeros((n, num_out_depth, num_out_row, num_out_col))
		for batch in range(n):
			for r in range(num_out_row):
				for c in range(num_out_col):
					patch_now = X[batch,:,self.stride*r:self.stride*r+self.filter_row,self.stride*c:self.stride*c+self.filter_col]
					ans[batch,:,r,c] = np.sum(np.sum(patch_now,1),1)/(self.filter_col*self.filter_row)

		return ans
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError
		new_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))
		
		def indices(in_ind, row):
			# max_index, rem, quo, min_index = 0
			if row == True:
				max_index = min(int(in_ind/self.stride), self.out_row-1)
				rem = (in_ind - self.filter_row) % self.stride
				quo = (in_ind - self.filter_row) / self.stride
			else:
				max_index = min(int(in_ind/self.stride), self.out_col-1)
				rem = (in_ind - self.filter_col) % self.stride
				quo = (in_ind - self.filter_col) / self.stride

			if rem == 0:
				min_index = int(max(0, quo + 1))
			else:
				min_index = int(max(0, np.ceil(quo)))
			return min_index, max_index

		for batch in range(n):
			for r in range(self.in_row):
				for c in range(self.in_col):
					r_min, r_max = indices(r, True)
					c_min, c_max = indices(c, False)
					for i in range(r_min, r_max+1):
						for j in range(c_min,c_max+1):
							new_delta[batch,:,r,c] += (delta[batch,:,i,j]/(self.filter_row*self.filter_col))

		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Function for the activation and its derivative
def relu_of_X(X):

	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu

	ans = X.copy()
	ans[ans < 0] = 0.0

	return ans
	# raise NotImplementedError
	
def gradient_relu_of_X(X, delta):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu amd during backwardpass

	ans = X.astype(float)
	ans[ans < 0] = 0.0
	ans[ans > 0] = 1.0
	return ans

def softmax_of_X(X):
	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax
	
	# raise NotImplementedError
	ans = X.astype(float)
	for i in range(X.shape[0]):
		tot = np.exp(X[i])
		totsum = np.sum(tot)
		ans[i] = tot/totsum
	return ans
	
def gradient_softmax_of_X(X, delta):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax amd during backwardpass
	# Hint: You might need to compute Jacobian first

	ans = X.astype(float)
	m,n = X.shape

	for i in range(m):
		jacob = np.zeros((n,n), dtype = float)
		for j in range(n):
			for k in range(n):
				if j == k:
					jacob[j][k] = (X[i][j]) * (1 - X[i][j])
				else:
					jacob[j][k] = -(X[i][j]) *  X[i][k]

		ans[i] = (jacob @ delta[i]).reshape((n,))

	return ans