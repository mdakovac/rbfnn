# math lib
import numpy as np

# plotting lib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ml algorithms
from sklearn.cluster import KMeans
import rbfnn as rbfnn


# sample inputs and add noise
NUM_SAMPLES = 800
X = np.random.uniform(0, 1.5, NUM_SAMPLES)
# print(X)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = 3*np.sin(12*X) + noise
# y = 15*X + nois#e
X = X.reshape(-1, 1)


# K-fold splitting
X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)


# cluster range
min_clusters = 2
max_clusters = 15
cluster_range = range(min_clusters, max_clusters+1)

validation_MSEs, test_MSEs = rbfnn.analyze(X_train_validate, X_test, y_train_validate, y_test, min_clusters, max_clusters, "an", 1, 1, 1)


# plot
plt.plot(cluster_range, validation_MSEs, label="validation")
plt.plot(cluster_range, test_MSEs, label="testing")
plt.xticks(cluster_range)

plt.suptitle("sine", fontsize=12)
plt.xlabel("Number of kernels")
plt.ylabel("MSE")

plt.legend()
plt.show()

'''
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters).fit(X)
#print(kmeans.cluster_centers_)
c, s = rbfnn.prepare_data(X, kmeans.labels_, n_clusters, single_std=0)

#print(c)
#print(s)
# print(kmeans.labels_)

# rbfnn
q = 1
scaleStd = False
if scaleStd:
	for i in range(0, len(s)):
		s[i] = float(s[i])*q


nn = rbfnn.RBFNN(k=n_clusters, c=c, s=s)
nn.train(X, y, method="an")

# print("Clusters: " + str(n_clusters) + ", MSE: " + str(nn.get_MSE(X,y)))
y_pred = nn.predict(X)

plt.plot(X, y, '-o', label='true')
plt.plot(np.sort(X), y_pred, '-o', label='RBFNN')
plt.legend()

# plt.tight_layout()
plt.show()
'''
