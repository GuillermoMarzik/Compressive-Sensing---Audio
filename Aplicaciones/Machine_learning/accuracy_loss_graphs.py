#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:31:43 2021

@author: gmarzik
"""

import numpy as np
import matplotlib.pyplot as plt
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

loss_original = np.load('loss_original.npy')
loss_dct = np.load('loss_dct_50.npy')
accuracy_original = np.load('accuracy_original.npy')
accuracy_dct = np.load('accuracy_dct_50.npy')

fig,ax = plt.subplots(figsize=(7,5)  , dpi=80)
lns1 = ax.plot(accuracy_original, color='tab:blue',label='Exactitud')
ax.set_xlabel("Lotes")
ax.set_ylabel("Exactitud")
ax2=ax.twinx()
lns2 = ax2.plot(loss_original,color='tab:orange',label='Función de costo')
ax2.set_ylabel("Función de costo")
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right')
plt.savefig('acc_loss_orig.pgf')

fig,ax = plt.subplots(figsize=(7,5)  , dpi=80)
lns1 = ax.plot(accuracy_dct, color='tab:blue',label='Exactitud')
ax.set_xlabel("Lotes")
ax.set_ylabel("Exactitud")
ax2=ax.twinx()
lns2 = ax2.plot(loss_dct,color='tab:orange',label='Función de costo')
ax2.set_ylabel("Función de costo")
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right')
plt.savefig('acc_loss_dct.pgf')

fig,ax = plt.subplots(figsize=(7,5)  , dpi=80)
lns1 = ax.plot(accuracy_original, color='tab:blue',label='Exactitud - original')
ax.set_xlabel("Lotes")
ax.set_ylabel("Exactitud")
lns3 = ax.plot(accuracy_dct, color='tab:green',label='Exactitud - comprimido')
ax2=ax.twinx()
lns2 = ax2.plot(loss_original,color='tab:orange',label='Función de costo - original')
lns4 = ax2.plot(loss_dct,color='tab:olive',label='Función de costo - comprimido')
ax2.set_ylabel("Función de costo")
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right')
plt.savefig('acc_loss_global.pgf')