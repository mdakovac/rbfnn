# math lib
import numpy as np

# dataframe handling lib
import pandas as pd

# plotting lib
import matplotlib.pyplot as plt

from sklearn import preprocessing


# read data
df = pd.read_excel('results/airfoil_noise/single_std=1--random_centers=0.xlsx')

x = np.array(df[df.columns[0]])
q1 = np.array(df[df.columns[1]])
q12 = np.array(df[df.columns[2]])
q14 = np.array(df[df.columns[3]])
q16 = np.array(df[df.columns[4]])
q18 = np.array(df[df.columns[5]])
q2 = np.array(df[df.columns[6]])

'''
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, .05))
q1 = min_max_scaler.fit_transform(q1.reshape(-1, 1))
q1 = q1.reshape(1, -1)[0]

q2 = min_max_scaler.fit_transform(q2.reshape(-1, 1))
q2 = q2.reshape(1, -1)[0]

q3 = min_max_scaler.fit_transform(q3.reshape(-1, 1))
q3 = q3.reshape(1, -1)[0]

q4 = min_max_scaler.fit_transform(q4.reshape(-1, 1))
q4 = q4.reshape(1, -1)[0]

q5 = min_max_scaler.fit_transform(q5.reshape(-1, 1))
q5 = q5.reshape(1, -1)[0]

q6 = min_max_scaler.fit_transform(q6.reshape(-1, 1))
q6 = q6.reshape(1, -1)[0]

lambd = 0.5 * 0.001
for i in range(2, 21):
	q1[i-2] = q1[i-2] + lambd * i
	q2[i-2] = q2[i-2] + lambd * i
	q3[i-2] = q3[i-2] + lambd * i
	q4[i-2] = q4[i-2] + lambd * i
	q5[i-2] = q5[i-2] + lambd * i
	q6[i-2] = q6[i-2] + lambd * i

'''

# plot
plt.plot(x, q1, label="q=1.0")
plt.plot(x, q12, label="q=1.2")
plt.plot(x, q14, label="q=1.4")
plt.plot(x, q16, label="q=1.6")
plt.plot(x, q18, label="q=1.8")
plt.plot(x, q2, label="q=2.0")



plt.xticks(x)
plt.ylim(0, 1000)

plt.suptitle("Airfoil Self-Noise", fontsize=12)
plt.xlabel("Veličina")
plt.ylabel("MSE")

plt.legend()
plt.show()

