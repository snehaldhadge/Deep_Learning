# Deep learning using Boston dataset

import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Get Data from keras dataset
(X_train,y_train),(X_valid,y_valid) = boston_housing.load_data()

# Printing the shape of the Train and Validation Data Data
print(X_train.shape)
print(X_valid.shape)


model = Sequential()
model.add(Dense(32,activation='relu',input_dim=13))
model.add(BatchNormalization())

# Second Layer (Add dropout 20%)
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output is a Numerical value so Linear
model.add(Dense(1,activation='linear'))

print(model.summary())

# Here accuracy is not so important but how off out prediction
model.compile(loss='mean_squared_error',optimizer='adam')

# Finding when model performed well in each epoch
output_dir = 'model_output/'

run_name = 'regression_baseline'
output_path = output_dir+run_name


if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save out model weights (not bias so save_weights_only)

modelcheckpt = ModelCheckpoint(output_path,'/weights.{epoch:02d}.hdf5',
save_weights_only=True)


model.fit(X_train,y_train,batch_size=8,epochs=32,verbose=1,validation_data=(X_valid,y_valid),callbacks=[modelcheckpt])

# Choose which was best
model.load_weights(output_path+'/weights.28.hdf5')

X_valid[42]

model.predict(np.reshape(X_valid[42],[1,13]))

# %%
# Adding tensorboard
from tensorflow.keras.callbacks import TensorBoard
# Get Data from keras dataset
(X_train,y_train),(X_valid,y_valid) = boston_housing.load_data()

# Printing the shape of the Train and Validation Data Data
print(X_train.shape)
print(X_valid.shape)


model = Sequential()
model.add(Dense(32,activation='relu',input_dim=13))
model.add(BatchNormalization())

# Second Layer (Add dropout 20%)
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output is a Numerical value so Linear
model.add(Dense(1,activation='linear'))

print(model.summary())

# Here accuracy is not so important but how off out prediction
model.compile(loss='mean_squared_error',optimizer='adam')

# Finding when model performed well in each epoch
output_dir = 'model_output/'

run_name = 'regression_baseline'
output_path = output_dir+run_name


if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save out model weights (not bias so save_weights_only)

modelcheckpt = ModelCheckpoint(output_path,'/weights.{epoch:02d}.hdf5',
save_weights_only=True)

tensorboard = TensorBoard(log_dir='logs/'+run_name)

model.fit(X_train,y_train,batch_size=8,epochs=32,verbose=1,validation_data=(X_valid,y_valid),callbacks=[modelcheckpt,tensorboard])

# Choose which was best
model.load_weights(output_path+'/weights.28.hdf5')

X_valid[42]

model.predict(np.reshape(X_valid[42],[1,13]))

# Then go to the terminal and run
# tensorboard --logdir='logs/' --port 6006
# takes you to website
# %%
