# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:53:00 2020

@author: kasy
"""

import numpy as np
import _pickle as cp

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(all_data, all_target, test_size=0.3, random_state=42)
#y_train=np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)

from keras.callbacks import ModelCheckpoint, TensorBoard

import os

## val 0.852, test 0.874
## possion loss, weight decay 1e-4
log_dir = './model/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


hl_data = np.load('./data/health.npy')
pa_data = np.load('./data/pa.npy')

hl_data = np.asarray(hl_data)
hl_label = np.zeros(len(hl_data))

pa_data = np.asarray(pa_data)
pa_label = np.ones(len(pa_data))

all_data = np.concatenate([hl_data, pa_data], axis=0)
all_label = np.concatenate([hl_label, pa_label], axis=0)

## all_data normalize
all_mean = np.mean(all_data)
all_std = np.std(all_data)
all_data = (all_data -all_mean)/all_std

num, step = all_data.shape
#for i in range(num):
#    mean_i = np.mean(all_data[i, :])
#    std_i = np.std(all_data[i, :])
#    
#    if std_i > 5e-2:
#        all_data[i, :] = (all_data[i, :] - mean_i)/std_i
#    print('*********std  :',mean_i)
    
#max_all = np.max(all_data)    
#min_all = np.min(all_data)
#all_data = (all_data - min_all)/(max_all - min_all)
    
print(np.max(all_data))
all_data = all_data[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label,\
                                                    test_size=0.4, random_state=20)

y_train=np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

val_num = int(X_test.shape[0])
val_x = X_test[:val_num//2, ...]
val_y = y_test[:val_num//2, ...]

X_test = X_test[val_num//2:, ...]
y_test = y_test[val_num//2:, ...]

print(y_train.shape)

##===========参数=================
NB_SENSOR_CHANNELS = 1 #通道数目
NUM_CLASSES = 2       #类别数
BATCH_SIZE = 16
NUM_FILTERS = 64
FILTER_SIZE = 3
NUM_UNITS_LSTM = 64
STRIDE_SIZE=2 #卷积步长

time_step = 600

EPOCH_NUM = 250


##====================1====================
rmp = optimizers.RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=1e-4)
#rmp = optimizers.Adam(learning_rate=5e-4)

model = Sequential()
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, strides=STRIDE_SIZE,activation='relu',
                 input_shape=(time_step, NB_SENSOR_CHANNELS), 
                 name='Conv1D_1'))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', 
                 name='Conv1D_2'))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', 
                 name='Conv1D_3'))
#model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, strides=STRIDE_SIZE, activation='relu', kernel_initializer='orthogonal',
#                 name='Conv1D_4'))

model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, 
               name='LSTM_1'))
model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, 
               name='LSTM_2'))
model.add(Flatten(name='Flatten'))
model.add(Dropout(0.8, name='dropout'))
model.add(Dense(NUM_CLASSES, activation='softmax', 
                name='Output'))

#model.compile(loss='categorical_crossentropy', optimizer=rmp, metrics=['accuracy'])

model.compile(loss='poisson', \
              optimizer=rmp, metrics=['accuracy'])


print(model.summary())

##===========================2===============

checkpoint = ModelCheckpoint(log_dir + "best_weights.h5",
                                 monitor="val_accuracy",
                                 mode='max',
                                 save_weights_only=True,
                                 save_best_only=True, 
                                 verbose=1,
                                 period=1)

tensorboard = TensorBoard(log_dir=log_dir)
call_backs = [tensorboard, checkpoint]

model.fit(X_train, y_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE,verbose=1,\
          validation_data=(val_x, val_y), callbacks=call_backs)

#from keras.models import load_model
model.load_weights('./model/best_weights.h5')
import time
start =time.clock()
test_pred = np.argmax(model.predict(X_test), axis=1)
test_true = np.argmax(y_test, axis=1)
#np.unique(test_pred)
import sklearn.metrics as metrics
print("Test accauracy:\t{:.4f} ".format(metrics.accuracy_score(test_true, test_pred)))
end = time.clock()
#print('Running time: %s Seconds'%(end-start))

