# -*- coding: utf-8 -*-

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

#Store data in a dataframe
import pandas as pd
dataset = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
dataset['progress'] = diabetes.target

# Split data into X and y
X = dataset.iloc[:, 0:10]
y = dataset.iloc[:, -1]

#Split data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Train model using 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#Calculate Model Accuracy
print(lr.score(X_test, y_test))
