# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("titanic3.csv")
#delete the column which is not required
dataset.drop(['name', 'ticket', 'cabin', 'home.dest','boat','body'], axis=1, inplace=True)

X = dataset.drop(columns=['survived'])
y = dataset.survived

#dataset = dataset.dropna(subset=['embarked'])
print(dataset.isnull().sum(axis = 0))
print(dataset.dtypes)
pipeline_1 = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                             StandardScaler())
pipeline_2 = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                             OneHotEncoder())
preprocessor = make_column_transformer(
    (pipeline_1, ['age', 'fare']),
    (pipeline_2, ['sex', 'embarked']),
    remainder='passthrough'    
)
pipeline = make_pipeline(preprocessor, SelectKBest(k=3,score_func=f_classif), RandomForestClassifier())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_test, y_test))












