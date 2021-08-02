# -*- coding: utf-8 -*-


import pandas as pd

df = pd.read_csv("pima-data.csv")

#1. Null values
print(df.isnull().values.any())

#2. Correlation

corr = df.corr()
del df['skin']

#3. Data Molding
diab_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diab_map)

X = df.iloc[:, 0:8]
y = df.iloc[:, -1]
#4. Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

#5. Impute
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)