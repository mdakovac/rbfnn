# math lib
import numpy as np

# dataframe handling lib
import pandas as pd

# preprocessing
from sklearn import preprocessing

# plotting lib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ml algorithms
from sklearn.cluster import KMeans
import rbfnn as rbfnn

# cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut


df = pd.read_csv('data/computer_hardware.csv', header=None)
df = df[df.columns[2:10]]

# normalizacija
df = np.array(df)
df = preprocessing.normalize(df)
df = pd.DataFrame(df)

X = np.array(df[df.columns[0:7]])
y = np.array(df[df.columns[7]])
print(y)

X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)
MSEs = []
clusters = range(2, 25);
for i in range(2, 25):

	print("\n\nClusters: " + str(i))
	for q in range(1, 4):
		print("q = " + str(q), flush=True)
		kf = KFold(n_splits=10, shuffle=True)
		validation_MSEs = []

		for train_index, test_index in kf.split(X_train_validate):

			X_train, X_validate = X_train_validate[train_index], X_train_validate[test_index]
			y_train, y_validate = y_train_validate[train_index], y_train_validate[test_index]

			n_clusters = i
			kmeans = KMeans(n_clusters=n_clusters).fit(X_train)

			c,s = rbfnn.prepare_data(X_train, kmeans.labels_, n_clusters, single_std=False)

			for j in range(0,len(s)):
				s[j] = float(s[j])*q

			# rbfnn
			nn = rbfnn.RBFNN(k=len(c), c=c, s=s)
			nn.train(X_train, y_train, method="an")

			current_MSE = nn.get_MSE(X_validate, y_validate)

			validation_MSEs.append(current_MSE)

		print("K-Fold validation MSE: " + str(np.mean(validation_MSEs)), flush=True)
		MSEs.append(np.mean(validation_MSEs))

	# testiranje možda ne valja
	#print("Test subset MSE: " + str(nn.get_MSE(X_test, y_test)), flush=True)

c = np.array_split(MSEs,3)
print(c)

plt.plot(clusters, c[0], '-o', label='q=1')
plt.plot(clusters, c[1], '-o', label='q=2')
plt.plot(clusters, c[2], '-o', label='q=3')
plt.legend()
 
plt.xlabel("Clusters")
plt.ylabel("MSE")
plt.show()