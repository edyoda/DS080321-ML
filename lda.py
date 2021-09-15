# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("pima-data.csv")

del df['skin']
diabetes_map = {True:1, False:0}
df['diabetes'] = df['diabetes'].map(diabetes_map)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 0)

from sklearn.impute import SimpleImputer

fill_0 = SimpleImputer(missing_values=0, strategy='mean')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
print(lda.explained_variance_ratio_)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)
print(gnb.score(X_test, y_test))


