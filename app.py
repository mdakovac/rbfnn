# math lib
import numpy as np

# dataframe handling lib
import pandas as pd

# plotting lib
import matplotlib.pyplot as plt

# ml algorithms
import rbfnn as rbfnn

import os

# cross-validation
from sklearn.model_selection import train_test_split

import sys

validFile = False
while not validFile:
	try:
		filePath = input("Enter the filepath of desired dataset: ")
		header = input("Does your dataset have a header? (y/n): ")

		# ucitaj dataset
		if str(header) == "y":
			df = pd.read_csv(str(filePath)).astype(float)
		else:
			df = pd.read_csv(str(filePath), header=None).astype(float)

		validFile = True
	except:
		print("Invalid dataset. Please try again.")
		print("")

print("")
validInputRange = False
while not validInputRange:
	try:
		colRange = input("Enter input column range separated by a comma (first column is numbered 0), e.g. 0,6: ")
		colRange = colRange.split(',')
		startCol = int(colRange[0])
		if len(colRange) < 2:
			endCol = startCol
		else:
			endCol = int(colRange[1])

		validInputRange = True

	except:
		print("Invalid input. Try again.")

validOutputCol = False
while not validOutputCol:
	try:
		outputCol = input("Enter the column number for output (first column is numbered 0): ")
		outputCol = int(outputCol)
		validOutputCol = True
	except:
		print("Not a number. Try again.")

if startCol == endCol:
	df.drop_duplicates(subset=df.columns[startCol], inplace=True)

	# izdvoji ulaze i izlaze
	X = np.array(df[df.columns[startCol]])
	y = np.array(df[df.columns[outputCol]])

else:
	df.drop_duplicates(subset=df.columns[startCol:endCol], inplace=True)

	# izdvoji ulaze i izlaze
	X = np.array(df[df.columns[startCol:endCol]])
	y = np.array(df[df.columns[outputCol]])


print("--------------------------------------------------------")
print("Dataset import complete.")
print("--------------------------------------------------------")

validClusterRange = False
while not validClusterRange:
	try:
		inputColRange = input("Enter cluster number range separated by a comma, e.g. 2,20 or press Return to use [2,20]: ")
		if inputColRange == "":
			min_clusters = 2
			max_clusters = 20
		else:
			inputColRange = inputColRange.split(',')
			min_clusters = int(inputColRange[0])
			if len(inputColRange) < 2:
				max_clusters = min_clusters
			else:
				max_clusters = int(inputColRange[1])

		cluster_range = range(min_clusters, max_clusters + 1)
		validClusterRange = True
	except:
		print("Invalid input. Try again.")


validqRange = False
while not validqRange:
	try:
		qRange = input("Enter width factor q values separated by a comma, e.g. 1,1.2,1.4,1.6 or press Return to use [1,1.2,1.4,1.6,1.8,2]: ")
		if qRange == "":
			q = np.linspace(1, 2, 6)
		else:
			qRange = qRange.split(',')
			for i in range(0, len(qRange)):
				qRange[i] = float(qRange[i])
			q = np.array(qRange)

		validqRange = True
	except:
		print("Invalid input. Try again.")


km = input("Use K-means clustering? (y/n): ")
if km == "y":
	random_centers = 0
else:
	random_centers = 1

validExportPath = False
while not validExportPath:
	exportPath = input("Enter path for data export: ")
	if not os.path.isdir(exportPath):
		print("Invalid path. Please try again.")
	else:
		validExportPath = True

single_std = 0

print("--------------------------------------------------------")
print("Configuration finished. Starting...")
print("--------------------------------------------------------")
print("")

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

#plt.suptitle("Real Estate", fontsize=12)
plt.xlabel("Number of kernels")
plt.ylabel("MSE")

plt.legend()
plt.show()

# export to excel
excel_data = pd.DataFrame(output_object)
excel_data.to_excel(exportPath+'/'+'results--random_centers='+str(random_centers)+'.xlsx', sheet_name='sheet1', index=False)
