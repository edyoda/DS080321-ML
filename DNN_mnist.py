# -*- coding: utf-8 -*-

import pandas as pd
data = pd.read_csv("mnist-train.csv")

features = data.columns[1:]
X = data[features]
y = data['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X/255, y , test_size = 0.1, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
print(np.unique(y_train, return_counts=True))

n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

dnn = Sequential()
dnn.add(Dense(units = 100, kernel_initializer='uniform', activation = 'relu', input_shape = (784,)))
dnn.add(Dropout(0.2))

dnn.add(Dense(units = 10, kernel_initializer='uniform', activation = 'softmax'))

dnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = dnn.fit(X_train, y_train, batch_size = 64, epochs = 30, validation_data=(X_test, y_test))

y_pred = dnn.predict(X_test)
y_pred = (y_pred > 0.9)


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training_loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title("training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
















