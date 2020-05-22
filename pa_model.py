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

all_data = all_data[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label,\
                                                    test_size=0.3, random_state=42)

y_train=np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

##===========参数=================
NB_SENSOR_CHANNELS = 1 #通道数目
NUM_CLASSES = 2       #类别数
BATCH_SIZE = 16
NUM_FILTERS = 32
FILTER_SIZE = 5
NUM_UNITS_LSTM = 64
STRIDE_SIZE=2 #卷积步长

time_step = 600

EPOCH_NUM = 100


##====================1====================
rmp = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, strides=STRIDE_SIZE,activation='relu', kernel_initializer='orthogonal',
                 input_shape=(time_step, NB_SENSOR_CHANNELS), 
                 name='Conv1D_1'))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', kernel_initializer='orthogonal', 
                 name='Conv1D_2'))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', kernel_initializer='orthogonal', 
                 name='Conv1D_3'))
#model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, strides=STRIDE_SIZE, activation='relu', kernel_initializer='orthogonal',
#                 name='Conv1D_4'))
model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, kernel_initializer='orthogonal', 
               name='LSTM_1'))
model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, kernel_initializer='orthogonal', 
               name='LSTM_2'))
model.add(Flatten(name='Flatten'))
model.add(Dropout(0.8, name='dropout'))
model.add(Dense(NUM_CLASSES, activation='softmax', kernel_initializer='orthogonal', 
                name='Output'))

model.compile(loss='categorical_crossentropy', optimizer=rmp, metrics=['accuracy'])

print(model.summary())

##===========================2===============

model.fit(X_train, y_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE,verbose=1)


import time
start =time.clock()
test_pred = np.argmax(model.predict(X_test), axis=1)
test_true = np.argmax(y_test, axis=1)
#np.unique(test_pred)
import sklearn.metrics as metrics
print("\tTest accauracy:\t{:.4f} ".format(metrics.accuracy_score(test_true, test_pred)))
end = time.clock()
print('Running time: %s Seconds'%(end-start))

