# math lib
import numpy as np

# plotting lib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ml algorithms
from sklearn.cluster import KMeans
import rbfnn as rbfnn


# sample inputs and add noise
NUM_SAMPLES = 300
X = np.random.uniform(0, 2, NUM_SAMPLES)
#print(X)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = 3*np.sin(12*X) + noise
#y = 15*X + nois#e



n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters).fit(X.reshape(-1,1))

c,s = rbfnn.prepare_data(X, kmeans.labels_, n_clusters, single_std=True)

print(c)
print(kmeans.labels_)

# rbfnn
q = 3
scaleStd = False
if(scaleStd):
	for i in range(0,len(s)):
		s[i] = float(s[i])*q


nn = rbfnn.RBFNN(k=n_clusters, c=c, s=s)
nn.train(X,y, method="an")

print("Clusters: " + str(n_clusters) + ", MSE: " + str(nn.get_MSE(X,y)))
y_pred = nn.predict(X)




plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBFNN')
plt.legend()
 
#plt.tight_layout()
plt.show()