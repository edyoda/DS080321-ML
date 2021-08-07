# -*- coding: utf-8 -*-
##########################################################
#Use as Classification

import pandas as pd
dataset = pd.read_csv("Fish.csv")
#1. Find correlated feature and clean it
corr = dataset.corr()
del dataset['Length2']
del dataset['Length3']

#2. Store into X and y
X = dataset.iloc[:, 1:5]
y = dataset.iloc[:, 0]

# Apply label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#3. Split into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
#4. Standardize the values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#5. Train the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

#6. Calculate accuracy
print(knn.score(X_test, y_test))