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
df.drop_duplicates(subset=df.columns[0:6], inplace=True)


# izdvoji ulaze i izlaze
X = np.array(df[df.columns[0:6]])
y = np.array(df[df.columns[6]])


#cluster range
min_clusters = 2
max_clusters = 20
min_q = 1
max_q = 2
single_std = 1
random_centers = 1


cluster_range = range(min_clusters, max_clusters+1)
q = np.linspace(min_q, max_q, 6)

output_object = {'Kernels': cluster_range}
for q in q:
	#X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, random_state=random.randint(0, 100000), test_size=0.25)

	validation_MSEs = rbfnn.analyze(X,
								    y,
								    min_clusters,
								    max_clusters,
								    train_method="an",
								    q=q,
								    single_std=single_std,
								    random_centers=random_centers,
								    print_results=1)

	plt.plot(cluster_range, validation_MSEs, label="q = "+str(q))
	output_object["MSE, q="+str(q)] = validation_MSEs

# plot
#plt.plot(cluster_range, test_MSEs, label="testing")
plt.xticks(cluster_range)
#plt.ylim(0, 0.3)

plt.suptitle("Real Estate", fontsize=12)
plt.xlabel("Number of kernels")
plt.ylabel("MSE")

plt.legend()
plt.show()

# export to excel
excel_data = pd.DataFrame(output_object)
excel_data.to_excel('results/real_estate/single_std='+str(single_std)+'--random_centers='+str(random_centers)+'.xlsx', sheet_name='sheet1', index=False)
