import numpy
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
import os
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import tensorflow.python.ops.nn_ops
import pandas as pd
K.set_image_data_format('channels_last')
numpy.random.seed(0)

model = Sequential()

from keras.layers import Dropout

model.add(Conv2D(96, kernel_size=11, padding="same", input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(4, 4)))

model.add(Conv2D(96, kernel_size=11, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(4, 4)))

model.add(Conv2D(256, kernel_size=5, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(384, kernel_size=3, padding="same", activation = 'relu'))
model.add(Conv2D(384, kernel_size=3, padding="same", activation = 'relu'))
model.add(Conv2D(256, kernel_size=3, padding="same", activation = 'relu'))

from keras.layers.core import Activation

model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4096, activation='relu'))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])