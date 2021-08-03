# -*- coding: utf-8 -*-

import pandas as pd
exam_data = pd.read_csv("exams.csv")

#1. Data Standardization
from sklearn import preprocessing
exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])

#2. Label Encoding - Target
le = preprocessing.LabelEncoder()
exam_data['gender'] = le.fit_transform(exam_data['gender'])

#3. One Hot Encoding
exam_data = pd.get_dummies(exam_data, columns=['race/ethnicity',  'parental level of education',
                            'lunch', 'test preparation course'], drop_first=True)


X = exam_data.iloc[:, 1:15]
y = exam_data.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))


