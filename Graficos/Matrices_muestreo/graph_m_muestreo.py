#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:14:01 2021

@author: gmarzik
"""

import numpy as np
import pickle
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib
import tikzplotlib
import soundfile as sf
from scipy.fft import dct

matplotlib.style.use('default')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


x,fs = sf.read('musica.wav')
musica_f = dct(x)
len_x = len(x)
non_zero_coeffs = len(np.where(musica_f>0.1)[0])
muestras_teoricas = np.int(non_zero_coeffs*np.log(len_x/non_zero_coeffs))
p_m_teoricas = 100*(muestras_teoricas/len_x)

Xorig_2 = plt.imread('cerebro.jpeg')
len_x = Xorig_2.shape[0]*Xorig_2.shape[1]

def dct2(x):
    return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
X_f = dct2(Xorig_2)
non_zero_coeffs = len(np.where(X_f>1)[0])
muestras_teoricas = np.int(non_zero_coeffs*np.log(len_x/non_zero_coeffs))
p_m_teoricas = 100*(muestras_teoricas/len_x)

path = ['./Bernoulli/','./Gauss/','./Poisson/']
transf = 'dct'
m_muestreo = ['bernoulli','gauss','poisson']
#para señales temporales
param = ['rmse','lsd','snr','pesq','stoi','psnr','ssim']
#para imagen
#transformadas = ['dct','dwt','curvelet']

metrica = []
# for i in range(3):
#     infile = open(path[i]+transf+'_'+m_muestreo[i]+'_owlqn_'+param[3]+'.pickle','rb')
#     val = pickle.load(infile)
#     infile.close()
#     metrica.append(val[3])
    
#imagen
for i in range(3):
    infile = open(path[i]+'imagen_'+transf+'_'+m_muestreo[i]+'_owlqn_'+param[0]+'.pickle','rb')
    val = pickle.load(infile)
    infile.close()
    metrica.append(val)
    
    

p_muestras = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
matplotlib.rcParams.update({'font.size': 12})

fig,ax = plt.subplots(figsize=(7,5)  , dpi=80)
for i in range(len(metrica)):
    ax.plot(p_muestras,np.sqrt(metrica[i]), label=m_muestreo[i].capitalize())
ax.legend()
ax.set_xlabel('Porcentaje de muestras utilizadas para la reconstrucción [%]')
ax.set_ylabel('RMSE')
#ax.set_yscale('log')
#ax.set_yticks(ticks=(10,15,20,30,40))
# for axis in [ax.yaxis]:
#     formatter = FuncFormatter(lambda y, _: '{:.6g}'.format(y))
#     formatter = FuncFormatter(lambda y,_: np.format_float_positional(y, trim='-'))
#     axis.set_major_formatter(formatter)
#plt.show()
ax.set_xticks(ticks=p_muestras)
ax.axvline(x=p_m_teoricas,color='red',linestyle='--')
#Sweep ticks RMSE
#ax.set_yticks(ticks=[0.05,0.1,0.2,0.4,0.8])
#Sweep ticks LSD
#ax.set_yticks(ticks=[5,10,15,20,25,30,40,50,60])
#Imagen ticks RMSE
#ax.set_yticks(ticks=[0.05,0.1,0.2,0.4,0.8])
ax.grid(which='major',axis='both',c='grey',lw=0.000000001)
plt.savefig('imagen_rmse_m_muestreo.pgf')
