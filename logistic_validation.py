# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("pima-data.csv")

corr = df.corr()
del df['skin']

#data molding
diab_map = { True : 1, False: 0}
df['diabetes'] = df['diabetes'].map(diab_map)
X = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values = 0, strategy = 'mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.7 , max_iter=140)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV

inter = [100, 120, 140, 160]
cvlaue = [0.2, 0.4, 0.6, 0.8, 1.2, 1.4]

pa_grid = dict(C=cvlaue, max_iter=inter)
lr_grid = LogisticRegression(penalty ='l2')
grid = GridSearchCV(estimator=lr_grid , param_grid=pa_grid, cv = 5)
grid_result = grid.fit(X_train, y_train)
print(grid_result.best_score_)
print(grid_result.best_params_)








