# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv("HR.csv")

#Store in X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

#Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Standardise the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Apply Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C= 0.8, max_iter=120)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))


#Apply Grid Search
#max_iter, C
from sklearn.model_selection import GridSearchCV
iters = [100, 120, 140, 160]
Cv = [0.7, 1, 1.2, 1.4]
params = dict(max_iter=iters, C=Cv)
lr_gs = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr_gs, param_grid=params, cv =5)

grid_result = grid.fit(X_train, y_train)
print(grid_result.best_score_)
print(grid_result.best_params_)


############################################################################
# Pipeline

from sklearn.pipeline import make_pipeline

pipeline_1 = make_pipeline(StandardScaler(), LogisticRegression())
pipeline_1.fit(X_train, y_train)
y_pred = pipeline_1.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print(pipeline_1.steps[1][1].coef_)


# Select 3 Best features among 7 features
from sklearn.feature_selection import SelectKBest, f_classif
pipeline_2 = make_pipeline(StandardScaler(),SelectKBest(k=3, score_func=f_classif),
                           LogisticRegression(C = 0.8, max_iter = 100))


print(pipeline_2)
pipeline_2.fit(X_train, y_train)
y_pred = pipeline_2.predict(X_test)

print(accuracy_score(y_test, y_pred))


params = {'selectkbest__k': [3, 4, 5, 6],
          'logisticregression__C': [0.7, 1, 1.2, 1.4],
          'logisticregression__max_iter' : [100, 120, 140, 160]
    }

gs = GridSearchCV(pipeline_2, param_grid=params, cv = 5)
gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)





















