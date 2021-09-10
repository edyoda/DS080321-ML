# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("BankNote_Authentication.csv")

X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators = 200, max_depth = 50, learning_rate = 0.01)

gb.fit(X_train, y_train)
print(gb.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
a = [100, 120, 140, 160, 180, 200]
b = [20, 30, 40, 50]
c = [0.01, 0.05, 0.1]

param_grid = dict(n_estimators = a, max_depth = b, learning_rate = c)
gb_c= GradientBoostingClassifier()
grid = GridSearchCV(estimator = gb, param_grid = param_grid, cv= 4)

grid_result = grid.fit(X_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)