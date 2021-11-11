#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:21:35 2021

@author: gmarzik
"""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import matplotlib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


matplotlib.style.use('default')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

modelo = tf.keras.models.load_model('./Modelos_entrenados/modelo_sym8_50')
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full_m = np.zeros((60000,32,32))
X_test_full = np.zeros((10000,32,32))
X_train_full_m[:,2:-2,2:-2] = X_train_full
X_test_full[:,2:-2,2:-2] = X_test

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

y_pred = modelo.predict_classes(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision,recall,f_score,support = precision_recall_fscore_support(y_test,y_pred)
precision = np.mean(precision)
recall = np.mean(recall)
f_score = np.mean(f_score)
