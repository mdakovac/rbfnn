# math lib
import numpy as np

# dataframe handling lib
import pandas as pd

# plotting lib
import matplotlib.pyplot as plt

# ml algorithms
import rbfnn as rbfnn

# cross-validation
from sklearn.model_selection import train_test_split

import sys

# ucitaj dataset
df = pd.read_csv('data/real_estate_valuation_data.csv').astype(float)


# izdvoji ulaze i izlaze
# ignore cols 4 and 5, all same values
X = np.array(df[df.columns[0:6]])
y = np.array(df[df.columns[6]])

# cluster range
min_clusters = 2
max_clusters = 20
cluster_range = range(min_clusters, max_clusters+1)

X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)
validation_MSEs1, test_MSEs1 = rbfnn.analyze(X_train_validate,
										   X_test,
										   y_train_validate,
										   y_test,
										   min_clusters,
										   max_clusters,
										   train_method="an",
										   single_std=1,
										   normalize=1,
										   print_results=0)

X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.25)
validation_MSEs2, test_MSEs2 = rbfnn.analyze(X_train_validate,
										   X_test,
										   y_train_validate,
										   y_test,
										   min_clusters,
										   max_clusters,
										   train_method="an",
										   single_std=0,
										   normalize=1,
										   print_results=0)

# plot
plt.plot(cluster_range, validation_MSEs1, label="single")
plt.plot(cluster_range, validation_MSEs2, label="different")
plt.xticks(cluster_range)

plt.suptitle("Real Estate", fontsize=12)
plt.xlabel("Number of kernels")
plt.ylabel("MSE")

plt.legend()
plt.show()

# export to excel
excel_data = pd.DataFrame({'Kernels': cluster_range,
						   's validation MSE': validation_MSEs1,
						   's testing MSE': test_MSEs1,
						   'd validation MSE': validation_MSEs2,
						   'd testing MSE': test_MSEs2})
excel_data.to_excel('results/test.xlsx', sheet_name='sheet1', index=False)

