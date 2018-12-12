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


df = pd.read_csv('data/airfoil_self_noise.csv', header=None)

# normalizacija
df = np.array(df)
df = preprocessing.normalize(df)
df = pd.DataFrame(df)

X = np.array(df[df.columns[0:2]])
y = np.array(df[df.columns[5]])
print(y)

X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)

MSEs, ca = rbfnn.analyze(X_train_validate, X_test, y_train_validate, y_test, 2, 25, "an", False)

rbfnn.plot(	x = ca, 
			y = MSEs, 
			x_label="Number of kernels", 
			y_label="MSE", 
			title="Airfoil noise with multiple standard deviation")