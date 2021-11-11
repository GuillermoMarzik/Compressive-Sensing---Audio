#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:28:27 2021

@author: gmarzik
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from rec_cnn import subsamp, cs_cnn
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
np.save('X_train_full_orig.npy',X_train_full)
np.save('y_train_full_orig.npy',y_train_full)
np.save('X_test_orig.npy',X_test)
np.save('y_test_orig.npy',y_test)

imag = X_train_full[1,:,:]
imag_m = np.zeros([32,32],dtype='uint8')
imag_m[2:-2,2:-2] = imag
ri,b = subsamp(imag_m,0.5)
Xa = cs_cnn(ri,b)

plt.figure(1)
plt.imshow(imag_m,cmap='binary')
plt.figure(2)
plt.imshow(Xa,cmap='binary')




