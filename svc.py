# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("BankNote_Authentication.csv")

#X = dataset.iloc[:, 0:4].values
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.svm import SVC
svc = SVC(kernel = 'poly', random_state= 0, C = 0.1)
svc.fit(X_train, y_train)

print(svc.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
param = {'C': [0.1, 0.5, 1, 5 , 10],
         'kernel': ['linear', 'rbf']}

gs = GridSearchCV(svc, param_grid=param, cv = 5)
gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)

#Re usable Graph Code
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
clf = svc 
h = 0.1
X_plot, z_plot = X_train, y_train

x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.4, cmap = ListedColormap(('blue', 'red')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM - Polynomial")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()