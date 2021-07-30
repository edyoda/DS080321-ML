# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, 2:5].values

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 6, init = 'k-means++', random_state = 0)
k_means.fit(X)

print(k_means.labels_)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters = k, init = 'k-means++')
    km.fit(X)
    wcss_i = km.inertia_
    wcss.append(wcss_i)
    print(k,wcss_i)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()