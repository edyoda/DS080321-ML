# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("Sentiment.csv")

dataset.candidate.value_counts().plot(kind='pie', autopct = '%1.0f%%')

twit_sent = dataset.groupby(['candidate', 'sentiment']).sentiment.count().unstack()
twit_sent.plot(kind='bar')

dataset = dataset.drop(dataset[dataset.sentiment == 'Neutral'].index)
sent_map = {"Positive": 1, "Negative":0}
dataset['sentiment'] = dataset['sentiment'].map(sent_map)

X = dataset["text"]
y = dataset["sentiment"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Vectorizing
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.7, min_df = 1, stop_words='english')

trainX = tfidf.fit_transform(X_train).toarray()
testX = tfidf.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(trainX, y_train)

print(gnb.score(testX, y_test))

mnb = MultinomialNB()
mnb.fit(trainX, y_train)
print(mnb.score(testX, y_test))


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(trainX, y_train)

print(model.score(testX, y_test))











