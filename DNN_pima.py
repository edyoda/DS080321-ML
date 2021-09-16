# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv("pima-data.csv")

del df['skin']
diab_map = {True:1, False:0}
df['diabetes'] = df['diabetes'].map(diab_map)

X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.impute import SimpleImputer

fill_0 = SimpleImputer(missing_values=0, strategy = 'mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

dnn = Sequential()

dnn.add(Dense(units = 8, kernel_initializer='uniform', activation = 'relu', input_dim = 8))

dnn.add(Dense(units = 5, kernel_initializer='uniform', activation = 'relu'))

dnn.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))

dnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

dnn.fit(X_train, y_train, batch_size = 8, epochs = 75)

y_pred = dnn.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
























