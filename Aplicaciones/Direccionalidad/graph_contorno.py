#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:57:20 2021

@author: gmarzik
"""

import numpy as np

import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from graph_contorno_funcs import load_rec
from Suavizado import filtropromediomovil
import soundfile as sf



finverso,fs = sf.read('./inverseFilter800-20k5sec.wav')

#Cargo los archivos
Z = load_rec('curvelet',0.5)


#defino variables para los gráficos
r = 200
l = 48000+r
m = np.arange(l)
T = ((48000+r)/fs)
frec = m/T
frec = frec[750:20000]
DriverSinBocina_rec = np.zeros([48000+r,37])        

#calculo respuestas al impulso
for a in range(37):
    dataimp = sig.fftconvolve(Z[:,a],finverso); #saco la rta al impulso del barrido que cargo
    dataimp = dataimp[np.argmax(dataimp) - r:np.argmax(dataimp) + 48000] # Corto el impulso desde el maximo 1 segundo para atras y 1,8 ms para adelante.
    DriverSinBocina_rec[:,a] = dataimp

#cálculo del espectro y pasaje a dB
DriverSinBocina_rec_frec = np.fft.fft(DriverSinBocina_rec.T)[:,750:20000]
Driver_rec_frec_log = 20*np.log10(np.abs(DriverSinBocina_rec_frec))    

#suavizado por octavos de octava        
Rec_frec_log = np.zeros((37,19250))

for n in range(37):
    Rec_frec_log[n,:] = filtropromediomovil(Driver_rec_frec_log[n,:],frec,8)

#normalización con respecto a la medición en el eje
norm_rec = Rec_frec_log[0,:]
Rec_frec_log_norm = np.zeros((37,19250))
for n in range(37):
    Rec_frec_log_norm[n,:] = Rec_frec_log[n,:] - norm_rec  
    
#definición de la malla
ang = np.array(np.arange(-180,185,5))
frec = np.array(frec)
X,Y = np.meshgrid(frec,ang)
Rec_frec_log_norm = np.float64(Rec_frec_log_norm)
lvls = np.linspace(-45,-0,16)
contorno_rec = np.zeros([73,len(Rec_frec_log_norm[0,:])])

#se replican las mediciones con respecto al eje central
contorno_rec[0:37][:] = Rec_frec_log_norm[::-1][:]
contorno_rec[37:][:] = Rec_frec_log_norm[1:][:]
for i in range(19250):   
    for c in range(73):
        if contorno_rec[c][i] >0:
            contorno_rec[c][i] = 0 

#gráficos de contorno a partir de las mallas
fig3 = plt.figure(2)
ax3 = plt.subplot(111)
cs = ax3.contourf(X,Y,contorno_rec,levels = lvls, cmap='jet')
ax3.set_ylabel('Ángulo respecto al eje')
ax3.set_xlabel('Frecuencia [Hz]')
cbar = plt.colorbar(cs,label='dB')
ax3.set_xscale('log')
ax3.tick_params(axis='x',which='minor',bottom=False)
ax3.xaxis.set_major_formatter(ScalarFormatter())
ax3.set_xticks([1000,2000,4000,8000,16000])
tick = np.arange(-180,185,30)
tick = tick[::-1]
ax3.set_yticks(tick)
