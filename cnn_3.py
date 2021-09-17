# -*- coding: utf-8 -*-

# Dataset - https://drive.google.com/drive/folders/13teCT5fs0mAnecMpazrKsllPJhAOas8w


# CNN

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten

cnn = Sequential()

# Step 1 - Convolution

cnn.add(Convolution2D(32, (3,3), input_shape=(128, 128, 3), activation = 'relu'))

# Step -2 Maxpooling
cnn.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flatten
cnn.add(Flatten())

#Step 4 - Full Connection
cnn.add(Dense(units = 128, activation='relu'))
cnn.add(Dense(units = 3, activation = 'softmax'))

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Step 5 - Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_gen = ImageDataGenerator(rescale = 1./255)

train_data = train_gen.flow_from_directory('dataset_3/dataset_3/train', target_size = (128, 128),
                                           batch_size = 64, class_mode = 'categorical')
test_data = test_gen.flow_from_directory('dataset_3/dataset_3/val', target_size = (128, 128),
                                         batch_size = 64, class_mode = 'categorical')

# training and test 
cnn.fit_generator(train_data, steps_per_epoch = 40,
                  epochs = 20, validation_data=test_data)



















