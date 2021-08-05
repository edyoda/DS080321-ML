# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("Company_Performance.csv")
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

#Fitting Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
print(linear_reg.score(X, y))

#Visualize Simple Linear Regression
import matplotlib.pyplot as plt

plt.scatter(X, y, color = 'red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Size of Company(Simple Linear Regression)')
plt.xlabel('No of years')
plt.ylabel('No. of emp')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_pred = lin_reg_poly.predict(X_poly)
print(lin_reg_poly.score(X_poly, y))


#Visualization Polynomial LR
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Size of Company(Polynomial Linear Regression)')
plt.xlabel('No of years')
plt.ylabel('No. of emp')
plt.show()








