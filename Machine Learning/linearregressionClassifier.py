import argparse
import pandas as pd
import numpy as np


def linreg(features, labels):
    features_t = np.transpose(features)
    A = features_t@features
    e, v = np.linalg.eig(A)
    D = np.diag(e)
    for x in range(len(D)):
        if not D[x,x] == 0:
            D[x,x] = 1/D[x,x]
    A_plus = v@D@np.transpose(v)
    b = features_t@labels
    return A_plus@b

#get training data
#parse file
parser = argparse.ArgumentParser(description='Processes a data file')
parser.add_argument('--dataset', action='store', dest='file_name', required=True)
args = parser.parse_args()

#read file and convert to dataframe
df = pd.read_csv(args.file_name)
df.insert(loc=4, column='bias', value=1.0)
prediction_inputs = df.iloc[:, :5].to_numpy()
actual = df.iloc[:, -1].to_numpy()

#split dataset into 2
benign_df = df[df.iloc[:, -1] == 0]
benign_features = benign_df.iloc[:, :5]
benign_labels = benign_df.iloc[:, 5]
np_benign_features = benign_features.to_numpy()
np_benign_labels = benign_labels.to_numpy()

malignant_df = df[df.iloc[:, -1] == 1]
malignant_features = malignant_df.iloc[:, :5]
malignant_labels = malignant_df.iloc[:, 5]
np_malignant_features = malignant_features.to_numpy()
np_malignant_labels = malignant_labels.to_numpy()

benign_weights = linreg(np_benign_features, np_benign_labels)
malignant_weights = linreg(np_malignant_features, np_malignant_labels)

predicted_labels = []
for i in range(len(actual)):
    b_dist = abs(np.dot(prediction_inputs[i], benign_weights))/np.linalg.norm(benign_weights)
    m_dist = abs(np.dot(prediction_inputs[i], malignant_weights))/np.linalg.norm(malignant_weights)
    if b_dist > m_dist:
        predicted_labels.append(0)
    else:
        predicted_labels.append(1)


num_correct = 0
for i in range(len(actual)):
    if predicted_labels[i] == actual[i]:
        num_correct += 1

print("Accuracy: ")
print(num_correct/len(actual))


