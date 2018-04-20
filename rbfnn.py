import numpy as np 

class RBFNN:
	def __init__(self, k, c, s, epochs=100, learning_rate=0.01):
		# broj clustera tj. bases tj. gaussovih krivulja
		self.k = k

		# centroidi clustera
		self.c = c

		# standardna devijacija clustera
		self.s = s

		# broj epoha za treniranje
		self.epochs = epochs

		# learning rate
		self.learning_rate = learning_rate

		# lista s matricama čiji su članovi svi weights u NN
		# za pocetak su to random brojevi
		np.random.seed(0)
		self.weights = np.random.rand(1,k)

		# lista s matricama čiji su članovi svi biases u NN
		self.biases = np.random.rand(1, 1)

	def feedforward(self, input): 
		# input je lista, transponiranjem se dobije matrica n sa 1
		inputs = np.matrix(input)

		rbf_outputs = []
		for i in range(0, self.k):
			rbf_outputs.append(rbf(inputs, self.c[i], self.s[i]))

		rbf_outputs = np.transpose(np.matrix(rbf_outputs))

		net_output = np.add(np.matmul(self.weights, rbf_outputs), self.biases)

		outputs = [rbf_outputs, net_output]
		return outputs

	def train(self, input_list, target_list):
		for j in range(0, self.epochs):
			for i in range(0, len(input_list)):
				outputs = self.feedforward(input_list[i])
				target = target_list[i]

				error = -(target-outputs[1])

				delta_w = self.learning_rate * error[0, 0] * np.transpose(outputs[0])
				delta_b = self.learning_rate * error[0, 0]

				self.weights = np.subtract(self.weights, delta_w)
				self.biases = np.subtract(self.biases, delta_b)

	def set_learning_rate(self, lr):
		self.learning_rate = lr

	def predict(self, inputs):		
		y = []
		for i in range(0, len(inputs)):
			current_output = self.feedforward(inputs[i])
			y.append(current_output[1][0,0])

		return y

# activation function
def rbf(x, c, s):
	x = np.matrix(x)
	c = np.matrix(c)
	# euclidean distance: numpy.linalg.norm(a-b)
	return np.exp(-(np.square(np.linalg.norm(x-c))/(2*np.square(s))))

def prepare_data(inputs, labels, num_clusters, return_inputs=False):
	a = []
	c = []
	s = []
	for i in range(0, num_clusters):
		a.append([])

	for i in range(0, len(inputs)):
		a[labels[i]].append(inputs[i])


	distance_sum = 0
	current_mean = []
	for i in range(0, num_clusters):
		distance_sum = 0
		current_mean = np.mean(a[i], axis=0)

		for j in range(0, len(a[i])):
			distance_sum += np.square(np.linalg.norm(a[i][j]-current_mean))

		s.append(np.sqrt(distance_sum/len(a[i])))
		c.append(current_mean)

	if return_inputs:
		return c,s,a
	else:
		return c,s