#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:00:57 2021

@author: gmarzik
"""

import numpy as np
import pywt
from pylbfgs import owlqn
import os, os.path
import scipy.fftpack as spfft

def subsamp(imag,s):
    ny,nx = imag.shape
    k = round(nx * ny * s)
    ri = np.random.choice(nx * ny, k, replace=False).astype('uint16') # random sample of indices
    X_c=np.copy(imag)
    b = X_c.T.flat[ri].astype(float)
    b = X_c.T.flat[ri]
    return ri,b   

def subsamp_orig(imag):
    ny,nx = imag.shape
    ri = np.array(range(1024),dtype='uint16')
    b = imag.flatten()
    return ri,b
    

def cs_cnn(ri,b):
    nx = 32
    ny = 32
    
    def dct2(x):
        return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

    def idct2(x):
        return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    def evaluate(x, g, step):
        """An in-memory evaluation callback."""


        x2 = x.reshape((nx, ny)).T

        # x3 = pywt.array_to_coeffs(x2,coff_shape,output_format='wavedec2')
        # Ax2 = pywt.waverec2(x3,'sym8',mode='periodization')
        Ax2 = idct2(x2)
        

        Ax = Ax2.T.flat[ri].reshape(b.shape)

        Axb = Ax - b
        fx = np.sum(np.power(Axb, 2))

        Axb2 = np.zeros(x2.shape)
        Axb2.T.flat[ri] = Axb # fill columns-first

        # AtAxb2 = pywt.wavedec2(Axb2,'sym8',mode='periodization')
        # AtAxb2 = 2 * pywt.coeffs_to_array(AtAxb2)[0]
        AtAxb2 = 2 * dct2(Axb2)
        AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

        np.copyto(g, AtAxb)
    

        return fx
    def progress(x, g, fx, xnorm, gnorm, step, k, ls):
        #print('Iteration {}'.format(k))
        return 0

    aux = np.zeros((nx,ny))
    aux_w = pywt.wavedec2(aux,'sym8',mode='periodization')
    global coff_shape
    _,coff_shape= pywt.coeffs_to_array(aux_w)
    del aux, aux_w
    #Xat2 = owlqn(nx*ny, evaluate, progress, 0.001)
    Xat2 = owlqn(nx*ny, evaluate, None, 10)
    Xat = Xat2.reshape(nx, ny)
    # Xat = pywt.array_to_coeffs(Xat,coff_shape,output_format='wavedec2')
    # Xa = pywt.waverec2(Xat,'sym8',mode='periodization')
    Xa = idct2(Xat)
    Xa1 = Xa.T.clip(min=0)
    return Xa1
    
def load_dataset(path):
   tot_len = len(os.listdir(path))
   dataset = []
   dataset_instance = []
   for i in range(tot_len):
       with open(path+str(i)+'.npy', 'rb') as f:
           ri = np.load(f)
           b1 = np.load(f)
           b2 = np.load(f)
       dataset_instance.append(ri)
       dataset_instance.append(b1)
       dataset_instance.append(b2)
       dataset.append(dataset_instance)
       dataset_instance = []
   return dataset
