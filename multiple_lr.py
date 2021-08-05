# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("Largecap_Balancesheet.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Country", OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))
