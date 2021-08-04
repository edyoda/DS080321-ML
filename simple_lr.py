# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("Company_Profit.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
print(lr.predict(X_test))

#Visualization

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, lr.predict(X_train), color ='red')
plt.title("training set ")
plt.xlabel("Startup yrs operation")
plt.ylabel("Profit")
plt.show()


#Test Set
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, lr.predict(X_train), color ='red')
plt.title("test set ")
plt.xlabel("Startup yrs operation")
plt.ylabel("Profit")
plt.show()

















