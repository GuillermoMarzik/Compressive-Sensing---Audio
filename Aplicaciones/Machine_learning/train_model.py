#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:54:40 2021

@author: gmarzik
"""

import sys
import tensorflow as tf
import time
from rec_cnn import load_dataset
import keras
import numpy as np
from data_generator import build_generators
from modelo import clasificador

data = load_dataset('./Datasets/Original/Train/')

params = {'dataset': data, 'batch_size' : 4}
training_generator, validation_generator = build_generators(params)

model = clasificador()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))
        
cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),LossHistory()]

model.summary()

model.fit(training_generator,
                      validation_data =  validation_generator,
                      use_multiprocessing = True,
                      workers=12, epochs=10,
                      callbacks=cbks)
model.save('./Modelos_entrenados/modelo_original')
np.save('./Loss_history/loss_history_original.npy',cbks[1].history)