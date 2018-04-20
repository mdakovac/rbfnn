import rbfnn as rbfnn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
import rbfnn as rbfnn

'''
# prepare data
# -------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('C:/Users/matej/Desktop/Machine Learning/_RBFNN/data/auto-mpg.csv', header=None, na_values='?')
df = df[[0,3,4]].astype(float)
df.dropna(how="any", inplace=True)

X = np.array(df[[3,4]])
z = np.array(df[0])

xs = []
ys = []
for i in range(len(X)):
	xs.append(X[i][0])
	ys.append(X[i][1])


# use nn
# -------------------------------------------------------------------------------------------------------------------
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters).fit(X)
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)
c,s = rbfnn.prepare_data(X, kmeans.labels_, n_clusters)

nn = rbfnn.RBFNN(n_clusters, c, s, 100)
nn.set_learning_rate(0.03)
nn.train(X,z)
z_pred = nn.predict(X)


# plot
# -------------------------------------------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, z)
ax.scatter(xs, ys, z_pred)

ax.set_xlabel('HP')
ax.set_ylabel('Weight')
ax.set_zlabel('MPG')

plt.show()

'''
# sample inputs and add noise
NUM_SAMPLES = 100
X = np.random.uniform(0, 1.5, NUM_SAMPLES)
#print(X)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = -10*np.sin(2 * np.pi * X)  + noise

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=0).fit(X.reshape(-1,1))

c,s, X_new = rbfnn.prepare_data(X, kmeans.labels_, n_clusters, True)

# rbfnn
nn = rbfnn.RBFNN(n_clusters, c, s, 200)
nn.set_learning_rate(0.02)
nn.train(X,y)
y_pred = nn.predict(X)



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# plot
for i in range(0, len(X_new)):
	plt.plot(X_new[i], y, '-o', label='true', color = colors[i])


#plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBFNN')
plt.legend()
 
#plt.tight_layout()
plt.show()

#print(rbf(2, 3, 2))
