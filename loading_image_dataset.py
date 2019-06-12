import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

#Load Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalize dataset to 0-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Convert Class Vectors to binary class matrices
#Our labels are single values from 0 to 9
#Instead,, we want each label to be a array with one element set to 1 and the rest set to 0
