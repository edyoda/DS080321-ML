# -*- coding: utf-8 -*-
import pandas as pd
dataset = pd.read_csv("horror-train.csv")

#Store in X and y
X = dataset.text
y = dataset.author
#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=0)

#Apply countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print("Gaussian Naive Bayes Accuracy: {0}".format(gnb.score(X_test, y_test)))

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print("Multinomial Naive Bayes Accuracy: {0}".format(mnb.score(X_test, y_test)))

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
print("Bernoulli Naive Bayes Accuracy: {0}".format(bnb.score(X_test, y_test)))