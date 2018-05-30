import numpy as np 

# todo
# - mogućnost da se koristi svuda ista standardna devijacija
# - određivanje pogreške na testnom setu preko MSE za različite brojeve clustera, podjele datasetova, 
#   metode određivanja težina, metode određivanja standardne devijacije, learning_rateove
# - podjela dataseta na TT i TVT




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
		self.biases = [[0]]

	def feedforward(self, input): 
		# input je lista, transponiranjem se dobije matrica n sa 1
		inputs = np.matrix(input)

		rbf_outputs = []

		if(len(self.s) > 1):
			for i in range(0, self.k):
				rbf_outputs.append(rbf(inputs, self.c[i], self.s[i]))
		else:
			for i in range(0, self.k):
				rbf_outputs.append(rbf(inputs, self.c[i], self.s[0]))

		rbf_outputs = np.transpose(np.matrix(rbf_outputs))

		net_output = np.add(np.matmul(self.weights, rbf_outputs), self.biases)

		outputs = [rbf_outputs, net_output]
		return outputs

	def train(self, input_list, target_list, method="an"):
		if(method == "gd"):
			for j in range(0, self.epochs):
				for i in range(0, len(input_list)):
					outputs = self.feedforward(input_list[i])
					target = target_list[i]

					error = -(target-outputs[1])

					delta_w = self.learning_rate * error[0, 0] * np.transpose(outputs[0])
					delta_b = self.learning_rate * error[0, 0]

					self.weights = np.subtract(self.weights, delta_w)
					self.biases = np.subtract(self.biases, delta_b)

		elif(method == "an"):
			rbf_outputs = []
			for i in range(0, len(input_list)):
				outputs = self.feedforward(input_list[i])
				rbf_outputs.append(np.transpose(outputs[0]))

			rbf_outputs = np.concatenate(rbf_outputs)
			target_outputs = np.matrix(target_list)

			a_inv = np.linalg.inv(rbf_outputs.T * rbf_outputs)

			w = a_inv * rbf_outputs.T * target_outputs.T
			self.weights = w.T

		else:
			print("Unknown training method. Please specify bp or an.")
			

	def set_learning_rate(self, lr):
		self.learning_rate = lr

	def update_std_deviation(self, s):
		self.s = s

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
	return np.exp(-(np.square(np.linalg.norm(x-c))/(2*np.square(s))))

def prepare_data(inputs, labels, num_clusters, single_std=False):
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
		current_mean = np.mean(a[i], axis=0)
		c.append(current_mean)

		if(not single_std):
			distance_sum = 0

			for j in range(0, len(a[i])):
				distance_sum += np.square(np.linalg.norm(a[i][j]-current_mean))

			s.append(np.sqrt(distance_sum/len(a[i])))

	if(single_std):
		distances = []
		sm_distances = []
		for i in range(0, len(c)-1):
			for j in range(i+1, len(c)):
				distances.append(np.linalg.norm(c[i] - c[j]))
		
		for i in range(0, len(c)-1):
			smallest = min(distances)
			sm_distances.append(smallest)
			distances.remove(smallest)

		s = [max(sm_distances)/np.sqrt(2*len(c))]

	return c,s