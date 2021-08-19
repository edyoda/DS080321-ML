# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=0)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

print(dt.score(X_test, y_test))