# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
cancer = load_breast_cancer()
print(cancer.data.shape)
print(cancer.feature_names)
print(cancer.target)
pd.Series(cancer.data[:, 0]).plot.hist()

data = pd.DataFrame(cancer.data, columns= cancer.feature_names)
data['cancer'] = cancer.target

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
print("Gaussian Naive Bayes Accuracy: {0}".format(gnb.score(X_test, y_test)))

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print("Multinomial Naive Bayes Accuracy: {0}".format(mnb.score(X_test, y_test)))

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
print("Bernoulli Naive Bayes Accuracy: {0}".format(bnb.score(X_test, y_test)))
