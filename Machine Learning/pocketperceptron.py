import argparse
import pandas as pd
import numpy as np


#get training data

#parse file
parser = argparse.ArgumentParser(description='Processes a data file')
parser.add_argument('--dataset', action='store', dest='file_name', required=True)
args = parser.parse_args()

#read file and convert to dataframe
df = pd.read_csv(args.file_name)

#separate data into features and labels
features = df.iloc[:, :5]
labels = df.iloc[:, -1]

#convert pandas dataframes to numpy arrays
np_features = features.to_numpy()
np_labels = labels.to_numpy()
#replace 0 labels to -1
np.place(np_labels, np_labels == 0, -1)

#maximum number of steps
max_iter = 500


#initialization
#create zero weight vector
w = np.zeros(5)
#create pocket weight vector
w_pocket = np.zeros(5)
#run
run = 0
#run pocket
run_pocket = 0

#algo
updated = True
current_iter = 0
while updated and current_iter < max_iter:
    updated = False
    current_iter += 1
    for i in range(len(labels)):
        prediction = np.dot(w, np_features[i])
        y = np_labels[i]

        if (prediction >= 0 and y == -1) or (prediction < 0 and y == 1):
            updated = True
            w = np.add(w, np.multiply(np_features[i], np_labels[i]))
    label_predict = []
    for i in range(len(labels)):
        label_predict.append(np.sign(np.dot(w, np_features[i])))
    num_correct = 0
    for i in range(len(labels)):
        if label_predict[i] == np_labels[i]:
            num_correct += 1
    if num_correct > run:
        w_pocket = w
        run = num_correct

w = w_pocket

label_predict = []
for i in range(len(labels)):
    label_predict.append(np.sign(np.dot(w, np_features[i])))

num_correct = 0
for i in range(len(labels)):
    if label_predict[i] == np_labels[i]:
        num_correct += 1

print("Weight vector: ")
print(w)
