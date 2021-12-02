#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:05:14 2021

@author: gmarzik
"""

import numpy as np
from tkinter import filedialog
from tkinter import *
import scipy.signal as sig
from scipy.io import wavfile
from func_owl import subsamp,cs_rec
import scipy.fftpack as spfft

#%%
#Cargo archivos

#Directorio de los archivos
path = './Sweeps_grabados'
#Ángulo de rotación para las mediciones
Ang = 5 
sweeps = np.zeros([336000,37])

for a in range(0, 91, Ang):
    fs, data = wavfile.read(path + '/'+ str(a)+'.wav')
    nang = int(a/Ang) # paso del valor del angulo al indice del vector.
    sweeps[:,36-nang] = data[:336000,0]
    sweeps[:,nang] = data[:336000,1]
    del nang
sweeps = sweeps/np.max(sweeps)
print('Mediciones cargadas')

#%%
def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

#%%
#Reconstrucción

#Formateo variables de salida para la reconstrucción por DWT
Xorig = np.zeros([2**18,2**6])
Xorig[:,:37]= sweeps[50000:-23856,:] #Descarto ceros al principio y al final de los barridos frecuenciales para adaptarlos a la variable de salida
del sweeps

#Defino porcentajes de muestras a tomar en el proceso
sample_sizes=(0.8,0.3,0.5)

#DWT o DCT
wav = True

if wav:
    #Función wavelet
    wav_type = 'sym8'
    transformada = 'dwt'
    #Divido las reconstrucciones en dos grupos para mantener la potencia de dos
    Xorig1 = np.copy(Xorig[:,:32])
    Xorig2 = np.copy(Xorig[:,5:37])
    ny,nx = Xorig1.shape
    Z1 = [np.zeros(Xorig1.shape, dtype='float') for s in sample_sizes]
    Z2 = [np.zeros(Xorig2.shape, dtype='float') for s in sample_sizes]
    Z = [np.zeros(Xorig[:,:37].shape, dtype='float') for s in sample_sizes]
    del Xorig
else:
    transformada = 'dct'
    Xorig = np.copy(Xorig[:,:37])
    ny,nx = Xorig.shape
    Z = [np.zeros(Xorig.shape, dtype='float') for s in sample_sizes]



#Método híbrido o Compressive Sensing puro
CS = True
if CS:
    metodo = 'cs'
else:
    metodo = 'hibrido'
#Constante para OWL-QN
c = 0.01

for i,s in enumerate(sample_sizes):
    if wav:
        for j in range(2):
            if j==0:
                X_c = np.copy(Xorig1)
            else:
                X_c = np.copy(Xorig2)
            #Muestreo aleatorio
            #Primero selecciono los índices
            ri=subsamp(nx,ny,s,CS=CS)
            #Luego guardo el valor de las muestras conservadas
            b = X_c.T.flat[ri].astype(float)
            #Reconstruyo la señal a partir de las muestras del vector b
            #Atención: es lento
            Xat2 = cs_rec(ri,b,nx,ny,c,transform=transformada)
            #Antitransformo para volver al dominio temporal
            Xat = Xat2.reshape(nx, ny).T
            Xa = idct2(Xat)
            #Almaceno el resultado
            if j == 0:
                Z1[i][:,:] = np.copy(Xa)
            else:
                Z2[i][:,:] = np.copy(Xa)
        Z[i][:,:32] = np.copy(Z1[i][:,:])
        Z[i][:,32:] = np.copy(Z2[i][:,27:])
        #Guardo las reconstrucciones
        np.save('wav_sweeps_' + wav_type+'_'+metodo+'_'+str(s*100)+'.npy',Z[i])
    else:
        X_c = np.copy(Xorig)
        #Muestreo aleatorio
        #Primero selecciono los índices
        ri=subsamp(nx,ny,s,CS=CS)
        #Luego guardo el valor de las muestras conservadas
        b = X_c.T.flat[ri].astype(float)
        #Reconstruyo la señal a partir de las muestras del vector b
        #Atención: es lento
        Xat2 = cs_rec(ri,b,nx,ny,c,transform=transformada)
        #Antitransformo para volver al dominio temporal
        Xat = Xat2.reshape(nx, ny).T
        Xa = idct2(Xat)
        #Almaceno el resultado
        Z[i][:,:] = np.copy(Xa)
        #Guardo las reconstrucciones
        np.save('dct_sweeps_'+metodo+'_'+str(s*100)+'.npy',Z[i])