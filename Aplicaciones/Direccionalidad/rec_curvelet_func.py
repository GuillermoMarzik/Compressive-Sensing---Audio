#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:46:33 2021

@author: gmarzik
"""

import numpy as np
import pylops
from curvelops import FDCT2D
from pylops.optimization.sparsity import FISTA
import warnings
warnings.filterwarnings('ignore')
import os

def rec_curvelet(x,perc_subsampling,i):
    par = {}
    par['nx'], par['nt'] = x.shape
    par['dx'] = x[0, 1] - x[0,0]
    par['dt'] = x[1,0] - x[0,0]

    Nz, Nx = x.shape
    N = Nz * Nx

    Nsub2d = int(np.round(N*perc_subsampling))
    iava = np.sort(np.random.permutation(np.arange(N))[:Nsub2d])

    #Creo el operador de submuestreo
    Rop = pylops.Restriction(N, iava, dtype='float64')

    y = Rop*x.flatten()
    xadj = Rop.H*y.flatten()
    xadj = xadj.reshape(par['nt'], par['nx'])

    #Creo el operador de la transformada Curvelet
    DCTOp = FDCT2D((par['nx'], par['nt']))

    #del x[0]

    #Creo el operador que simultáneamente modeliza la curvelet y el submuestreo
    RCop = Rop*DCTOp.H
    
    #Reconstruyo la señal
    pl1, _, cost = FISTA(RCop, y.ravel(), niter=100, eps=1e-3, returninfo=True, show=True)
    
    #Antitransformo
    xl1 = np.real(DCTOp.H * pl1)

    #Vuelvo a las dimensiones originales
    xl1 = xl1.reshape(par['nx'], par['nt'])
    
    #Guardo las instancias
    np.save('curvelet_'+str(i)+'.npy',xl1)
    os._exit(0)