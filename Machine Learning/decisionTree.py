import argparse
import math
import pandas as pd
import numpy as np
import warnings

class Node:
    def __init__(self, category, value, length):
        self.children = []
        self.category = category
        self.value = value
        self.label = -1
        self.length = length


#returns index of feature with greatest information gain
def gain_idx(data):
    label = data[["Survived"]]
    label = label.to_numpy()
    features = data.drop(columns=["Survived"])
    features = features.to_numpy()


    #get number of features
    num_features = np.size(features, 1)

    #get number of data points
    num_datapoints = np.size(features,0)

    #create an array to store gains for each feature
    gain = [0]*num_features

    #computer entropy of label random variable
    num_passengers = np.size(label, 0)
    num_survived = np.count_nonzero(label == 1)
    num_died = num_passengers - num_survived

    pr_survived = num_survived/num_passengers
    pr_died = num_died/num_passengers

    label_entropy = -1*(pr_survived*math.log2(pr_survived)) + -1*(pr_died*math.log2(pr_died))

    #compute gain for each feature
    for i in range(num_features):

        table = {}

        for j in range(num_datapoints):
            if features[j,i] in table:
                table[features[j,i]][label[j,0]] += 1
            else:
                table[features[j, i]] = { 0:0, 1:0 }
                table[features[j, i]][label[j, 0]] += 1

        entropy_table = {}
        for value in table:
            count = table[value][0] + table[value][1]
            entropy_table[value] = { "pr_val":count/num_datapoints, "pr_0":table[value][0]/count, "pr_1":table[value][1]/count }

        feature_entropy = 0
        for value in entropy_table:
            pr_0 = entropy_table[value]["pr_0"]
            pr_1 = entropy_table[value]["pr_1"]
            if pr_0 == 1:
                feature_entropy += -1 * entropy_table[value]["pr_val"] * (pr_0 * math.log2(pr_0))
            elif pr_1 == 1:
                feature_entropy += -1 * entropy_table[value]["pr_val"] * (pr_1 * math.log2(pr_1))
            else:
                feature_entropy += -1*entropy_table[value]["pr_val"]*(pr_0*math.log2(pr_0) + pr_1*math.log2(pr_1))

        gain[i] = label_entropy-feature_entropy

    return gain.index(max(gain))


def compute_decision_tree(data, node, length):
    if len(data.columns) != 1:
        new_node = Node("label", None, node.length+1)
        np_data = data.to_numpy()
        label_dict = data["Survived"].value_counts().to_dict()
        if 0 in label_dict and len(data.index) == label_dict[0]:
            new_node.label = 0
            node.children.append(new_node)
        elif 1 in label_dict and len(data.index) == label_dict[1]:
            new_node.label = 1
            node.children.append(new_node)
        elif np.isclose(np_data, np_data[0]).all():
            new_node.label = data["Survived"].value_counts().idxmax()
            node.children.append(new_node)
        else:
            idx = gain_idx(data)
            col_values = data[data.columns[idx]].value_counts().index.tolist()
            if len(data.columns) > 2 and node.length < length:
                for value in col_values:
                    temp_node = Node(data.columns[idx], value, node.length+1)
                    new_data = data[data[data.columns[idx]] == value]
                    new_data.drop(new_data.columns[[idx]], axis=1, inplace=True)
                    compute_decision_tree(new_data, temp_node, length)
                    node.children.append(temp_node)
            else:
                for value in col_values:
                    temp_node = Node(data.columns[idx], value, node.length+1)
                    new_data = data[data[data.columns[idx]] == value]
                    l = new_data["Survived"].value_counts().idxmax()
                    label_node = Node("label", None, temp_node.length+1)
                    label_node.label = l
                    temp_node.children.append(label_node)
                    node.children.append(temp_node)


warnings.filterwarnings("ignore")

#get training data
#parse file
parser = argparse.ArgumentParser(description='Processes a data file')
parser.add_argument('--dataset', action='store', dest='file_name', required=True)
args = parser.parse_args()

#read file and convert to dataframe
df = pd.read_csv(args.file_name)
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df = df[pd.notnull(df["Survived"])]
df["Pclass"].fillna(df["Pclass"].value_counts().idxmax(), inplace=True)
df["Sex"].replace(to_replace=["male", "female"], value=[0, 1], inplace=True)
df["Sex"].fillna(df["Sex"].value_counts().idxmax(), inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Age"] = df["Age"].apply(lambda y: 0 if y < 30 else y)
df["Age"] = df["Age"].apply(lambda y: 1 if (y >= 30 and y < 45) else y)
df["Age"] = df["Age"].apply(lambda y: 2 if (y >= 45 and y < 64) else y)
df["Age"] = df["Age"].apply(lambda y: 3 if y >= 64 else y)
df["SibSp"].fillna(df["SibSp"].mean(), inplace=True)
df["Parch"].fillna(df["Parch"].mean(), inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Fare"] = df["Fare"].apply(lambda y: 0 if y < 7.8875 else 1)
df["Embarked"].replace(to_replace=["S", "C", "Q"], value=[0, 1, 2], inplace=True)
df["Embarked"].fillna(df["Embarked"].value_counts().idxmax(), inplace=True)

df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

msk = np.random.rand(len(df)) <= 0.6
train = df[msk]
test = df[~msk]

root = Node("root", None, 0)
compute_decision_tree(train, root, 6)

#Accuracy on training set
predicted_labels = []
for index, row in train.iterrows():
    current = root
    while current.children[0].category != "label":
        idx = None
        for i in range (len(current.children)):
            child = current.children[i]
            if row[child.category] == child.value:
                idx = i
            if idx == None:
                idx = 0
        current = current.children[idx]
    predicted_labels.append(current.children[0].label)

total = 0
correct = 0
idx = 0

for index, row in train.iterrows():
    total += 1
    if row["Survived"] == predicted_labels[idx]:
        correct += 1
    idx += 1

print("Training Set Accuracy: " + str(correct/total))

#Get accuracy on testing set
predicted_labels = []
for index, row in test.iterrows():
    current = root
    while current.children[0].category != "label":
        idx = None
        for i in range (len(current.children)):
            child = current.children[i]
            if row[child.category] == child.value:
                idx = i
            if idx == None:
                idx = 0
        current = current.children[idx]
    predicted_labels.append(current.children[0].label)

total = 0
correct = 0
idx = 0

for index, row in test.iterrows():
    total += 1
    if row["Survived"] == predicted_labels[idx]:
        correct += 1
    idx += 1

print("Testing Set Accuracy: " + str(correct/total))
