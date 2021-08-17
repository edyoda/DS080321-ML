# -*- coding: utf-8 -*-
import pandas as pd
hr_data = pd.read_csv("HR_comma_sep.csv")
X = hr_data.drop(columns=['left'])
y = hr_data.left

obj_data = X.select_dtypes(include=['object'])
float_data = X.select_dtypes(include=['float'])
int_data = X.select_dtypes(include=['int64'])

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

obj1_pipeline = make_pipeline(OrdinalEncoder())
obj2_pipeline = make_pipeline(OneHotEncoder())
int_pipeline = make_pipeline(MinMaxScaler(), SelectKBest(k=3, score_func=f_classif))
preprocessor = make_column_transformer(
                (obj1_pipeline, ['salary']),
                (obj2_pipeline, ['sales']),
                (int_pipeline, ['number_project', 'average_montly_hours', 'time_spend_company']),
                remainder = 'passthrough'
    )

master_pipeline = make_pipeline(preprocessor, RandomForestClassifier())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
master_pipeline.fit(X_train, y_train)
y_pred = master_pipeline.predict(X_test)
print(master_pipeline.score(X_test, y_test))
print(master_pipeline.steps[0][1].transformers)
print(master_pipeline)

from sklearn.model_selection import GridSearchCV
param = {'columntransformer__pipeline-3__selectkbest__k':[1,2,3]}
gs = GridSearchCV(master_pipeline, param_grid=param, cv = 5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)











