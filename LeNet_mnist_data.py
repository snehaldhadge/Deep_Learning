
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D 

# Loading the data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

print(X_train.shape)

# Pre process Data
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_valid = X_valid.reshape(10000, 28, 28, 1).astype('float32')

X_train /= 255
X_valid /= 255

# 10 digits 0-9
n_classes = 10

y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

# Model
model = Sequential()
# using 3 x 3
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a dropout of 25% at this layer
model.add(Dropout(0.25))
# Take 2 or 3 D activation array and flattens them to 1D array to pass them into dense layers
model.add(Flatten())
# Adiing a Dense layer with droupout of 50%
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# outputlayer
model.add(Dense(n_classes, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))

valid_0 = X_valid[0].reshape(1,784)
model.predict(valid_0)