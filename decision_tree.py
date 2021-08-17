# -*- coding: utf-8 -*-
import pandas as pd
dataset = pd.read_csv("tennis.csv")

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
X = dataset.drop(columns=['play'])
y = dataset.play
X = oe.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X, y)

print(dt.predict([[2, 1, 1, 10]]))