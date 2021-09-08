# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv("Largecap_Balancesheet.csv")

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

sfs_r = sfs(model, k_features = 3, forward = True, verbose =3, scoring = 'r2', cv = 4)

sfs_r = sfs_r.fit(X_train, y_train)

feature_sel = list(sfs_r.k_feature_idx_)
print(feature_sel)


model.fit(X_train[:, feature_sel], y_train)

print(model.score(X_test[:, feature_sel], y_test))