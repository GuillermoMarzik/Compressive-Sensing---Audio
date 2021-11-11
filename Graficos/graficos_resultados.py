#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:28:50 2021

@author: guillem
"""

import numpy as np
import pickle
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib
import tikzplotlib

matplotlib.style.use('default')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

path = './Habla/STOI/'
senal = 'habla'
param = 'stoi'
#para señales temporales
transformadas = ['dct','dft','dht','dwt','mdct','stdct','stft','stht']
#para imagen
# transformadas = ['dct','dwt','curvelet']
resultados = []
for i in range(len(transformadas)):        
    infile = open(path+senal+'_'+transformadas[i]+'_'+param+'.pickle','rb')
    val = pickle.load(infile)
    infile.close()
    resultados.append(val)

p_muestras = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]

#plt.style.use("ggplot")
matplotlib.rcParams.update({'font.size': 12})

color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:blue']
fig,ax = plt.subplots(figsize=(7,5)  , dpi=80)
#.upper()
for i in range(len(transformadas)):
    ax.plot(p_muestras,resultados[i], label=transformadas[i].upper(),color=color[i])
ax.legend()
ax.set_xlabel('Porcentaje de muestras utilizadas para la reconstrucción [%]')
ax.set_ylabel('STOI')
#ax.set_yscale('log')
for axis in [ax.xaxis, ax.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.15g}'.format(y))
    axis.set_major_formatter(formatter)
#plt.show()

ax.set_xticks(ticks=p_muestras)
ax.grid(which='both',axis='both',c='grey',lw=0.2)

plt.savefig('habla_stoi.pgf')
