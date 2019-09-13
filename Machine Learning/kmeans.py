import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("dota2Train.csv", header=None)
data = np.asanyarray(df)

kmeans = KMeans(n_clusters=2)
model = kmeans.fit(data)
clusters = model.labels_

win_counter = [0, 0]
lose_counter = [0, 0]

for i in range(len(data)):
    if data[i][0] == 1:
        win_counter[clusters[i]] += 1
    elif data[i][0] == -1:
        lose_counter[clusters[i]] += 1

print("Cluster 1: " + str(100 * win_counter[0]/sum(win_counter)) + "% of total wins.")
print("Cluster 1: " + str(100 * lose_counter[0]/sum(lose_counter)) + "% of total losses.")
print("Cluster 2: " + str(100 * win_counter[1]/sum(win_counter)) + "% of total wins.")
print("Cluster 2: " + str(100 * lose_counter[1]/sum(lose_counter)) + "% of total losses.")