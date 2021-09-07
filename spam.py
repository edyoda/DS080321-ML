# -*- coding: utf-8 -*-

import pandas as pd

dataset = pd.read_csv("emails.csv")

X = dataset['text']
y = dataset['spam']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
spam_fil = CountVectorizer(stop_words='english')

#from sklearn.feature_extraction.text import TfidfVectorizer
#spam_fil = TfidfVectorizer(max_df = 0.7, min_df = 1, stop_words='english')

trainX = spam_fil.fit_transform(X_train)
testX = spam_fil.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(trainX, y_train)
print(mnb.score(testX, y_test))

