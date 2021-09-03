# -*- coding: utf-8 -*-

# Import the dataset
import pandas as pd
df = pd.read_csv("pima-data.csv")

# Check for correlation
cor = df.corr()
del df['skin']
# Data Molding/Label Encoding
dia_dict = {True:1, False:0}
df['diabetes'] = df['diabetes'].map(dia_dict)

# Split into X and y
X = df.iloc[: , 0:8]
y = df.iloc[:, 8]


# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Imputing
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values = 0, strategy = 'mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)


# Apply Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))