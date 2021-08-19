# -*- coding: utf-8 -*-

import pandas as pd

dataset = pd.read_csv("Job_Exp.csv")

X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X, y)

y_pred = dt.predict([[39]])
print(y_pred)

import matplotlib.pyplot as plt
import numpy as np
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='green')
plt.plot(X_grid, dt.predict(X_grid), color = 'red')
plt.ylabel("Getting JOb Percentage")
plt.xlabel("Years of Exp")