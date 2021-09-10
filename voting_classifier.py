# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

estimator = [
    ('rf', RandomForestClassifier(n_estimators = 20)),
    ('svc', SVC(kernel = 'rbf', probability=True)),
    ('knc', KNeighborsClassifier()),
    ('abc', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=20)),
    ('lr', LogisticRegression())
    ]

vc = VotingClassifier(estimators = estimator, voting='hard')
vc.fit(X_train, y_train)
print(vc.score(X_test, y_test))

vc_s = VotingClassifier(estimators = estimator, voting='soft', weights = [ 5, 1, 1, 3, 4])
vc_s.fit(X_train, y_train)
print(vc_s.score(X_test, y_test))