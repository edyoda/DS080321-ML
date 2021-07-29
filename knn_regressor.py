# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names

df['House_Price'] = boston.target
print(df.isnull().values.any())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

nnr = KNeighborsRegressor(n_neighbors = 3)
nnr.fit(X_train, y_train)

print(nnr.score(X_test, y_test))

rmse_val = []
for K in range(1, 20):
    nn_model = KNeighborsRegressor(n_neighbors=K)
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    rmse_val.append(rmse)
    print("RMSE value=",rmse, "---K:", K)
    
    
curve = pd.DataFrame(rmse_val)
curve.plot()    