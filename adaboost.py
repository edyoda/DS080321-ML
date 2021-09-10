# -*- coding: utf-8 -*-

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression
adaboost = AdaBoostClassifier(n_estimators = 20, base_estimator = LogisticRegression())
adaboost.fit(X_train, y_train)
print(adaboost.score(X_test, y_test))