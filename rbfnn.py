import numpy as np 
import sys
# K-fold vs LOO vs random subsamples
# gradient descent vs analytical
# num of clusters
# std deviation
# learning rate

import random

# ml algorithms
from sklearn.cluster import KMeans

# cross-validation
from sklearn.model_selection import KFold

# dataframe handling lib
import pandas as pd

# preprocessing
from sklearn import preprocessing

# plotting lib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


class RBFNN:
	def __init__(self, k, c, s, epochs=200, learning_rate=0.01):
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
		self.weights = np.random.rand(1, k)

		# lista s matricama čiji su članovi svi biases u NN
		self.biases = [[0]]

	# activation function
	def rbf(self, x, c, s):
		x = np.matrix(x)
		c = np.matrix(c)
		return np.exp(-(np.square(np.linalg.norm(x - c)) / (2 * np.square(s))))

	# za jedan n-dimenzionalni input vraća outpute skrivenog i izlaznog sloja
	def feedforward(self, single_input):
		inputs = np.matrix(single_input)

		rbf_outputs = []

		if len(self.s) > 1:
			for i in range(0, self.k):
				rbf_outputs.append(self.rbf(inputs, self.c[i], self.s[i]))
		else:
			for i in range(0, self.k):
				rbf_outputs.append(self.rbf(inputs, self.c[i], self.s[0]))

		rbf_outputs = np.transpose(np.matrix(rbf_outputs))

		net_output = np.add(np.matmul(self.weights, rbf_outputs), self.biases)

		outputs = [rbf_outputs, net_output]
		return outputs

	# trening - update težina i biasa
	def train(self, input_list, target_list, method="an"):
		# gradient descent metoda
		if method == "gd":
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
		elif method == "an":
			rbf_outputs = []
			for i in range(0, len(input_list)):
				outputs = self.feedforward(input_list[i])
				rbf_outputs.append(np.transpose(outputs[0]))

			rbf_outputs = np.concatenate(rbf_outputs)
			target_outputs = np.matrix(target_list)

			# if matrix singular fall back to gradient descent
			try:
				a_inv = np.linalg.inv(rbf_outputs.T * rbf_outputs)
			except np.linalg.linalg.LinAlgError:
				print("Matrix singular - skipping", flush=True)
				# self.train(input_list, target_list, method="gd")
				return False

			w = a_inv * rbf_outputs.T * target_outputs.T
			self.weights = w.T

		return True

	# vraća n outputa mreže za n inputa
	def predict(self, input_list):	
		y = []
		for i in range(0, len(input_list)):
			current_output = self.feedforward(input_list[i])
			y.append(current_output[1][0, 0])
		return y

	# vraća Mean Squared Error za n inputa 
	def get_MSE(self, input_list, target_list):
		y = []
		for i in range(0, len(input_list)):
			current_output = self.feedforward(input_list[i])
			y.append(current_output[1][0, 0])

		mse = sum(np.square(target_list-y))/len(input_list)
		
		return mse


# pripremanje podataka za mrežu - izračun širina i koordinata središta kernela
def prepare_data(inputs, centers, single_std=False):
	a = []
	num_clusters = len(centers)
	stdds = []

	if not single_std:
		for i in range(0, num_clusters):
			p = 2
			neigh = NearestNeighbors(n_neighbors=p)
			neigh.fit(centers)
			distances = neigh.kneighbors(centers[i].reshape(1, -1))[0][0]
			stdds.append(np.sqrt(np.sum(np.square(distances)))/p)

	if single_std:
		'''
		# for sine test
		centers = np.sort(centers, axis=None)

		distances = []
		for i in range(0, len(centers)-1):
			distances.append(np.linalg.norm(centers[i]-centers[i+1]))

		stdds = [max(distances)/np.sqrt(2*num_clusters)]
		'''
		if num_clusters < 2:
			raise ValueError('Single standard deviation needs at least 2 kernels.')

		distances = []
		for i in range(0, len(centers)-1):
			for j in range(i+1, len(centers)):
				distances.append(np.linalg.norm(centers[i]-centers[j]))

		stdds = [max(distances)/np.sqrt(2*num_clusters)]

	return centers, stdds


def analyze(X_train_validate, X_test, y_train_validate, y_test, min_clusters, max_clusters, train_method, q, single_std=False, random_centers=False, normalize=True, print_results=True):
	validation_MSEs = []
	test_MSEs = []
	min_clusters = min_clusters
	max_clusters = max_clusters

	for i in range(min_clusters, max_clusters+1):
		if print_results:
			print("\n\nClusters: " + str(i))

		kf = KFold(n_splits=10, shuffle=True)
		KFold_validation_MSEs = []
		KFold_test_MSEs = []

		for train_index, test_index in kf.split(X_train_validate):
			X_train, X_validate = X_train_validate[train_index], X_train_validate[test_index]
			y_train, y_validate = y_train_validate[train_index], y_train_validate[test_index]

			if normalize:
				X_train, y_train = min_max_scale(X_train, y_train)
				X_validate, y_validate = min_max_scale(X_validate, y_validate)
				X_test, y_test = min_max_scale(X_test, y_test)

			n_clusters = i
			cluster_centers = []
			if not random_centers:
				kmeans = KMeans(n_clusters=n_clusters).fit(X_train)
				cluster_centers = kmeans.cluster_centers_
			else:
				for j in range(0, i):
					cluster_centers.append(random.choice(X_train))

			cluster_centers = np.array(cluster_centers)
			c, s = prepare_data(X_train, cluster_centers, single_std=single_std)

			s = np.multiply(q, s, dtype=float)

			nn = RBFNN(k=len(cluster_centers), c=cluster_centers, s=s)
			if not nn.train(X_train, y_train, method=train_method):
				continue

			KFold_validation_MSEs.append(nn.get_MSE(X_validate, y_validate))
			KFold_test_MSEs.append(nn.get_MSE(X_test, y_test))

		validation_MSEs.append(np.mean(KFold_validation_MSEs))
		test_MSEs.append(np.mean(KFold_test_MSEs))

		if print_results:
			print("K-Fold validation MSE: " + str(np.mean(KFold_validation_MSEs)), flush=True)
			print("K-Fold testing MSE: " + str(np.mean(KFold_test_MSEs)), flush=True)

	return validation_MSEs, test_MSEs


def min_max_scale(x, y):
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

	x = min_max_scaler.fit_transform(x)

	y = min_max_scaler.fit_transform(y.reshape(-1, 1))
	y = y.reshape(1, -1)[0]

	return x, y
	'''
	# normalizacija
	X_train = np.array(X_train)
	X_train = preprocessing.normalize(X_train)
	X_train = pd.DataFrame(X_train)
	'''

