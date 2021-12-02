#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 23:32:48 2021

@author: gmarzik
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim

orig = np.load('contorno_original.npy')

#híbrido
mat_hibrido = np.zeros((11,73,19250))
mat_hibrido[0,:,:] = np.load('contorno_dct_10_filas.npy')
mat_hibrido[1,:,:] = np.load('contorno_dct_30_filas.npy')
mat_hibrido[2,:,:] = np.load('contorno_dct_50_filas.npy')
mat_hibrido[3,:,:] = np.load('contorno_db4_10_filas.npy')
mat_hibrido[4,:,:] = np.load('contorno_db4_30_filas.npy')
mat_hibrido[5,:,:] = np.load('contorno_db4_50_filas.npy')
mat_hibrido[6,:,:] = np.load('contorno_db8_10_filas.npy')
mat_hibrido[7,:,:] = np.load('contorno_db8_30_filas.npy')
mat_hibrido[8,:,:] = np.load('contorno_db8_50_filas.npy')
mat_hibrido[9,:,:] = np.load('contorno_sym8_30_filas.npy')
mat_hibrido[10,:,:] = np.load('contorno_sym8_50_filas.npy')


#full cs
mat_full = np.zeros((15,73,19250))
mat_full[0,:,:] = np.load('contorno_dct_10.npy')
mat_full[1,:,:] = np.load('contorno_dct_30.npy')
mat_full[2,:,:] = np.load('contorno_dct_50.npy')
mat_full[3,:,:] = np.load('contorno_db4_10.npy')
mat_full[4,:,:] = np.load('contorno_db4_30.npy')
mat_full[5,:,:] = np.load('contorno_db4_50.npy')
mat_full[6,:,:] = np.load('contorno_db8_10.npy')
mat_full[7,:,:] = np.load('contorno_db8_30.npy')
mat_full[8,:,:] = np.load('contorno_db8_50.npy')
mat_full[9,:,:] = np.load('contorno_sym8_10.npy')
mat_full[10,:,:] = np.load('contorno_sym8_30.npy')
mat_full[11,:,:] = np.load('contorno_sym8_50.npy')
mat_full[12,:,:] = np.load('contorno_curvelet_10.npy')
mat_full[13,:,:] = np.load('contorno_curvelet_30.npy')
mat_full[14,:,:] = np.load('contorno_curvelet_50.npy')

#cálculo
ssim_hibrido = []
for i in range(11):
    ssim_hibrido.append(ssim(orig,mat_hibrido[i,:,:]))
    
ssim_full = []
for i in range(15):
    ssim_full.append(ssim(orig,mat_full[i,:,:]))