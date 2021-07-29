# -*- coding: utf-8 -*-

import pandas as pd

#import the dataset
dataset = pd.read_csv("50_Startups.csv")

#Split into X and y
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 4].values

# Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

#Fitting Nearest Neighbors Regression to the training set
from sklearn.neighbors import KNeighborsRegressor
nnr = KNeighborsRegressor(n_neighbors = 4)
nnr.fit(X_train, y_train)

#Calculate Score
print(nnr.score(X_test, y_test))


#Find the best value of K (n_neighbors)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_lst = []
for k in range(1, 20):
    nn_model = KNeighborsRegressor(n_neighbors = k)
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    error = sqrt(mean_squared_error(y_test, y_pred))
    rmse_lst.append(error)
    print(k, error)

#Draw the graph
graph = pd.DataFrame(rmse_lst)
graph.plot()