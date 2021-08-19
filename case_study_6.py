# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("titanic3.csv")

#Delete the columns which are not requried []
dataset.drop(['name', 'ticket', 'cabin', 'home.dest', 'boat', 'body'], axis=1, inplace=True)


#Store input and output in X and y
X = dataset.drop(columns=['survived'])
y = dataset.survived

print(dataset.isnull().sum(axis=0))

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
#Implement through pipeline
#pipeline_1 = age, fare - SimpleImputer , StandardScaler
#pipeline_2 = sex, embarked - SimpleImpputer, OnehoTencoder
pipeline_1 = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler())
pipeline_2 = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'), OneHotEncoder())

preprocessor = make_column_transformer(
    (pipeline_1, ['age', 'fare']),
    (pipeline_2, ['sex', 'embarked']),
    remainder = 'passthrough'
    )


master_pipeline = make_pipeline(preprocessor, SelectKBest(k=3, score_func=f_classif),DecisionTreeClassifier())


#split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)


#Train the model - DecisionTree
master_pipeline.fit(X_train, y_train)
#calculate the score
print(master_pipeline.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
param = {'selectkbest__k': [2, 3, 4,5,6]}
gs = GridSearchCV(master_pipeline, param_grid=param, cv=5)
gs.fit(X_train, y_train)
print(gs.best_score_)



