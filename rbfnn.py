import numpy as np 
import sys
# K-fold vs LOO vs random subsamples
# gradient descent vs analytical
# num of clusters
# std deviation
# learning rate


# ml algorithms
from sklearn.cluster import KMeans

# cross-validation
from sklearn.model_selection import KFold

# plotting lib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RBFNN:
	def __init__(self, k, c, s, epochs=500, learning_rate=0.01):
		# broj clustera tj. bases tj. gaussovih krivulja tj. kernels
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

	# za jedan n-dimenzionalni input vraća outpute skrivenog i izlaznog sloja
	def feedforward(self, single_input):
		inputs = np.matrix(single_input)

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

	# trening - update težina i biasa
	def train(self, input_list, target_list, method="an"):
		# gradient descent metoda
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

		# analitička metoda po
		# "Introduction to Radial Basis Function Networks", Mark J. L. Orr
		# Centre for Cognitive Science, University of Edinburgh, April 1996., poglavlje 4.1. 
		elif(method == "an"):
			rbf_outputs = []
			for i in range(0, len(input_list)):
				outputs = self.feedforward(input_list[i])
				rbf_outputs.append(np.transpose(outputs[0]))

			rbf_outputs = np.concatenate(rbf_outputs)
			target_outputs = np.matrix(target_list)

			try:
				a_inv = np.linalg.inv(rbf_outputs.T * rbf_outputs)

			except:
				#print(rbf_outputs.T * rbf_outputs)
				print("Matrix singular.", flush=True)
				return False

			w = a_inv * rbf_outputs.T * target_outputs.T
			self.weights = w.T
			return True
			#print(self.weights.astype(int))
			
	# vraća n outputa mreže za n inputa
	def predict(self, input_list):	
		y = []
		for i in range(0, len(input_list)):
			current_output = self.feedforward(input_list[i])
			y.append(current_output[1][0,0])
		return y

	# vraća Mean Squared Error za n inputa 
	def get_MSE(self, input_list, target_list):
		y = []
		for i in range(0, len(input_list)):
			current_output = self.feedforward(input_list[i])
			y.append(current_output[1][0,0])

		mse = sum(np.square(target_list-y))/len(input_list)
		
		return mse

	def set_learning_rate(self, lr):
		self.learning_rate = lr

	def update_std_deviation(self, s):
		self.s = s


# activation function
def rbf(x, c, s):
	x = np.matrix(x)
	c = np.matrix(c)
	return np.exp(-(np.square(np.linalg.norm(x-c))/(2*np.square(s))))

# pripremanje podataka za mrežu - izračun širina i koordinata središta kernela
def prepare_data(inputs, labels, num_clusters, single_std=False):
	a = []
	centers = []
	stdds = []
	for i in range(0, num_clusters):
		a.append([])

	for i in range(0, len(inputs)):
		a[labels[i]].append(inputs[i])

	a = np.array(a)
	for i in range(0, len(a)):
		centers.append(np.mean(a[i], axis=0))
		stdds.append(np.std(a[i], dtype='float64'))

	if(single_std):
		if(num_clusters<2):
			raise ValueError('Single standard deviation needs at least 2 kernels.')

		distances = []
		for i in range(0, len(centers)-1):
			for j in range(i+1, len(centers)):
				distances.append(np.linalg.norm(centers[i]-centers[j]))

		stdds = [max(distances)/np.sqrt(2*num_clusters)]

	return centers, stdds

def plot(x, y, x_label, y_label, title):
	plt.plot(x, y, '-o')
	plt.suptitle(title, fontsize=12)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def analyze(X_train_validate, X_test, y_train_validate, y_test, min_clusters, max_clusters, train_method, single_std):
	MSEs = []
	ca = []
	min_clusters = min_clusters
	max_clusters = max_clusters
	for i in range(min_clusters, max_clusters+1):
		print("\n\nClusters: " + str(i))

		kf = KFold(n_splits=10, shuffle=True)
		validation_MSEs = []

		for train_index, test_index in kf.split(X_train_validate):

			X_train, X_validate = X_train_validate[train_index], X_train_validate[test_index]
			y_train, y_validate = y_train_validate[train_index], y_train_validate[test_index]

			n_clusters = i
			kmeans = KMeans(n_clusters=n_clusters).fit(X_train)

			c,s = prepare_data(X_train, kmeans.labels_, n_clusters, single_std=single_std)

			# rbfnn
			nn = RBFNN(k=len(c), c=c, s=s)
			nn.train(X_train, y_train, method=train_method)

			current_MSE = nn.get_MSE(X_validate, y_validate)

			validation_MSEs.append(current_MSE)

		validation_MSE = np.mean(validation_MSEs)
		MSEs.append(validation_MSE)
		ca.append(i)

		if(validation_MSE > 5*np.mean(MSEs)):
			MSEs.pop()
			ca.pop()
			print("K-Fold validation MSE: " + str(validation_MSE), flush=True)
			return MSEs, ca

		print("10 puta mean: " + str(np.mean(MSEs)))
		print("K-Fold validation MSE: " + str(validation_MSE), flush=True)

	return MSEs, ca
