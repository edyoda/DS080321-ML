# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
auto_data = pd.read_csv("Detail_Cars.csv")

auto_data = auto_data.replace('?', np.nan)
col_object = auto_data.select_dtypes(include=['object'])

auto_data['price'] = pd.to_numeric(auto_data['price'], errors = 'coerce')
auto_data['bore'] = pd.to_numeric(auto_data['bore'], errors = 'coerce')
auto_data['stroke'] = pd.to_numeric(auto_data['stroke'], errors = 'coerce')
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors = 'coerce')
auto_data['peak-rpm'] = pd.to_numeric(auto_data['peak-rpm'], errors = 'coerce')

auto_data = auto_data.drop("normalized-losses", axis = 1)
cylin_map= {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
auto_data['num-of-cylinders'].replace(cylin_map, inplace=True)
auto_data = pd.get_dummies(auto_data, drop_first=True)
auto_data = auto_data.dropna()

X = auto_data.drop('price', axis = 1)
y = auto_data['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

from sklearn.linear_model import LinearRegression
li_model = LinearRegression()
li_model.fit(X_train, y_train)
print("Training Set:", li_model.score(X_train, y_train))
print("Test Set:",li_model.score(X_test, y_test))


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=5, normalize = True)
lasso.fit(X_train, y_train)
print("Training Set:", lasso.score(X_train, y_train))
print("Test Set:",lasso.score(X_test, y_test))

predictors = X_train.columns
coef = pd.Series(lasso.coef_, predictors).sort_values()
print(coef)

coef.plot(kind='bar')












