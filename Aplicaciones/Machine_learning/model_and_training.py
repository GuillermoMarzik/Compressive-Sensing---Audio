#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:15:36 2021

@author: gmarzik
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from functools import partial
from rec_cnn import cs_cnn

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full_m = np.zeros((60000,32,32))
X_test_full = np.zeros((10000,32,32))
X_train_full_m[:,2:-2,2:-2] = X_train_full
X_test_full[:,2:-2,2:-2] = X_test
X_train, X_valid = X_train_full_m[:-5000], X_train_full_m[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


X_train_full_m = np.zeros((60000,32,32),dtype='uint8')
y_train_full = np.zeros(60000)

path = './Datasets/50_porciento/Train/'

for i in range(len(X_train_full)):
    with open(path+str(i)+'.npy', 'rb') as f:
        ri = np.load(f)
        b = np.load(f)
        y_train_full[i] = np.load(f)
    X_train_full_m[i,:,:] = cs_cnn(ri,b)

X_train, X_valid = X_train_full_m[:-5000], X_train_full_m[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test_full - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]



DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[32, 32, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[],'accuracy':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))

cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),LossHistory()]

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid),callbacks=cbks)
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] 
y_pred = model.predict(X_new)

np.save('./Scores/score_sym8_50.npy',score)
model.save('./Modelos_entrenados/modelo_sym8_50')
np.save('./Loss_history/loss_history_sym8_50.npy',cbks[1].history)
