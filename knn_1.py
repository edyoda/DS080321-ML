# -*- coding: utf-8 -*-

import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

#Split into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))