# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
dataset = load_digits()
dataset.data.shape
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 0)

################
#Wihtout Pipeline
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
trainX = ss.fit_transform(X_train)
testX = ss.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(trainX, y_train)
rf.predict(testX)
print(rf.score(testX, y_test))


#***********************
#Pipeline

from sklearn.pipeline import make_pipeline
pipeline_1 = make_pipeline(StandardScaler(), RandomForestClassifier())

pipeline_1.fit(X_train, y_train)
y_pred = pipeline_1.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print(pipeline_1.steps[1][1].feature_importances_)

#############################################
#pipeline - Grid Search
from sklearn.feature_selection import SelectKBest, f_classif
pipeline_2 = make_pipeline(StandardScaler(), SelectKBest(k=10, score_func=f_classif),
                           RandomForestClassifier(n_estimators=100))
print(pipeline_2)
pipeline_2.fit(X_train, y_train)
y_pred = pipeline_2.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
params = { 'selectkbest__k': [10, 20, 30, 40],
          'randomforestclassifier__n_estimators': [100, 120, 140]
    }

gs = GridSearchCV(pipeline_2, param_grid=params, cv = 5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)

y_pred = gs.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_estimator_)






