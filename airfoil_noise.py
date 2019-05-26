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

import sys

# read data
df = pd.read_csv('data/airfoil_self_noise.csv', header=None)

# feature select
X = np.array(df[df.columns[0:3]])
y = np.array(df[df.columns[5]])

# K-fold splitting
X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)


# cluster range
min_clusters = 2
max_clusters = 20
cluster_range = range(min_clusters, max_clusters+1)

validation_MSEs, test_MSEs = rbfnn.analyze(X_train_validate, X_test, y_train_validate, y_test, min_clusters, max_clusters, "an", 1, 1, 1)


# plot
plt.plot(cluster_range, validation_MSEs, label="validation")
plt.plot(cluster_range, test_MSEs, label="testing")
plt.xticks(cluster_range)

plt.suptitle("Airfoil noise with multiple standard deviations", fontsize=12)
plt.xlabel("Number of kernels")
plt.ylabel("MSE")

plt.legend()
plt.show()


# export to excel
excel_data = pd.DataFrame({'Kernels': cluster_range, 'validation MSE': validation_MSEs, 'testing MSE': test_MSEs})
excel_data.to_excel('test.xlsx', sheet_name='sheet2', index=False)
