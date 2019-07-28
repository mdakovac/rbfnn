# math lib
import numpy as np

# plotting lib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ml algorithms
from sklearn.cluster import KMeans
import rbfnn as rbfnn

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

'''
# sample inputs and add noise
NUM_SAMPLES = 300
X = np.random.uniform(0, 1.5, NUM_SAMPLES)
# print(X)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.8, 0.8, NUM_SAMPLES)
y = 3*np.sin(12*X) + noise
# y = 15*X + nois#e
X = X.reshape(-1, 1)
'''
'''
# K-fold splitting
X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)


# cluster range
min_clusters = 2
max_clusters = 15
cluster_range = range(min_clusters, max_clusters+1)

validation_MSEs, test_MSEs = rbfnn.analyze(X_train_validate,
										   X_test,
										   y_train_validate,
										   y_test,
										   min_clusters,
										   max_clusters,
										   train_method="an",
										   single_std=1,
										   normalize=0,
										   print_results=1)


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
'''
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters).fit(X)

s = rbfnn.calculate_std(kmeans.cluster_centers_, single_std=0)

# rbfnn
q = 1
if q != 1:
	for i in range(0, len(s)):
		s[i] = float(s[i])*q

print(s)
nn = rbfnn.RBFNN(k=n_clusters, c=kmeans.cluster_centers_, s=s)
nn.train(X, y, method="an")

# print("Clusters: " + str(n_clusters) + ", MSE: " + str(nn.get_MSE(X,y)))
y_pred = nn.predict(X)

plt.plot(X, y, '-o', label='true')
plt.plot(np.sort(X), y_pred, '-o', label='RBFNN')
plt.legend()

# plt.tight_layout()
plt.show()
'''