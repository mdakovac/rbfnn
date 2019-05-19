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


# ucitaj dataset
df = pd.read_csv('data/energy_efficiency.csv').astype(float)

# normalizacija
df = np.array(df)
df = preprocessing.normalize(df)
df = pd.DataFrame(df)

# izdvoji ulaze i izlaze
X = np.array(df[df.columns[0:8]])
y = np.array(df[df.columns[9]])
print(y)

# razdvoji na setove na trening i testiranje
X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)

min_clusters = 2
max_clusters = 20
clusters = range(min_clusters, max_clusters+1)
qs = range(1,4)
for i in range(min_clusters, max_clusters+1):

	print("\n\nClusters: " + str(i))
	for q in qs:

		print("q = " + str(q), flush = True)

		kf = KFold(n_splits=10, shuffle=True)
		validation_MSEs = []

		for train_index, test_index in kf.split(X_train_validate):

			X_train, X_validate = X_train_validate[train_index], X_train_validate[test_index]
			y_train, y_validate = y_train_validate[train_index], y_train_validate[test_index]

			n_clusters = i
			kmeans = KMeans(n_clusters=n_clusters).fit(X_train)

			c,s = rbfnn.prepare_data(X_train, kmeans.labels_, n_clusters, single_std=False)

			for j in range(0, len(s)):
				s[j] = float(s[j])*q

			# rbfnn
			nn = rbfnn.RBFNN(k=len(c), c=c, s=s)
			if(not nn.train(X_train, y_train, method="an")):
				continue

			current_MSE = nn.get_MSE(X_validate, y_validate)
			
			validation_MSEs.append(current_MSE)

		print("K-Fold validation MSE: " + str(np.mean(validation_MSEs)), flush=True)


'''
MSEs, ca = rbfnn.analyze(X_train_validate, X_test, y_train_validate, y_test, 2, 15, "an", True)

rbfnn.plot( x=ca,
			y=MSEs,
			x_label="Number of kernels", 
			y_label="MSE", 
			title="Energy efficiency with single stdds")
'''