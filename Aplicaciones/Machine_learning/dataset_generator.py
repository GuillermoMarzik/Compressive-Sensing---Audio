#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:25:44 2021

@author: gmarzik
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from rec_cnn import subsamp, subsamp_orig


(X_train_prov, y_train), (X_test_prov, y_test) = keras.datasets.fashion_mnist.load_data()

X_train = np.zeros((60000,32,32),dtype='uint8')
X_test = np.zeros((10000,32,32))
X_train[:,2:-2,2:-2] = X_train_prov
X_test[:,2:-2,2:-2] = X_test_prov


path = './Datasets/Original/Train/'

for i in range(len(X_train)):
    ri,b = subsamp_orig(X_train[i,:,:])
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])
                
path = './Datasets/Original/Test/'

for i in range(len(X_test)):
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, X_test[i,:,:])
                np.save(f, y_test[i])

path = './Datasets/10_porciento/Train/'

for i in range(len(X_train)):
    ri,b = subsamp(X_train[i,:,:],0.1)
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])

path = './Datasets/20_porciento/Train/'

for i in range(len(X_train)):
    ri,b = subsamp(X_train[i,:,:],0.2)
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])

path = './Datasets/30_porciento/Train/'

for i in range(len(X_train)):
    ri,b = subsamp(X_train[i,:,:],0.3)
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])

path = './Datasets/40_porciento/Train/'

for i in range(len(X_train)):
    ri,b = subsamp(X_train[i,:,:],0.4)    
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])

path = './Datasets/50_porciento/Train/'

for i in range(len(X_train)):
    ri,b = subsamp(X_train[i,:,:],0.5)        
    with open(path+str(i)+'.npy', 'wb') as f:
                np.save(f, ri)
                np.save(f, b)
                np.save(f, y_train[i])
