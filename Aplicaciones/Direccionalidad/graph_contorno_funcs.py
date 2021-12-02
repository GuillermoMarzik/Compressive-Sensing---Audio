#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:58:33 2021

@author: gmarzik
"""

import numpy as np

def load_rec(transform,s,wav='db4',metodo='cs'):
    if transform == 'curvelet':
        if metodo == 'cs':
            if s == 0.1:
                sw1 = np.load('./Recs/curvelet_10_1.npy')
                sw2 = np.load('./Recs/curvelet_10_2.npy')
                sw3 = np.load('./Recs/curvelet_10_3.npy')
            elif s == 0.3:
                sw1 = np.load('./Recs/curvelet_30_1.npy')
                sw2 = np.load('./Recs/curvelet_30_2.npy')
                sw3 = np.load('./Recs/curvelet_30_3.npy')
            else:
                sw1 = np.load('./Recs/curvelet_50_1.npy')
                sw2 = np.load('./Recs/curvelet_50_2.npy')
                sw3 = np.load('./Recs/curvelet_50_3.npy')
        else:
            sw1 = np.load('./Recs/curvelet_50_1_lineas.npy')
            sw2 = np.load('./Recs/curvelet_50_2_lineas.npy')
            sw3 = np.load('./Recs/curvelet_50_3_lineas.npy')
        Z = np.hstack([sw1,sw2,sw3])
    elif transform == 'dwt':
        if metodo == 'cs':
            if wav == 'sym8':
                sw1 = np.load('./Recs/wav_sweeps_sym8_1_10_30_50.npy')
                sw2 = np.load('./Recs/wav_sweeps_sym8_2_10_30_50.npy')
            elif wav == 'db8':
                sw1 = np.load('./Recs/wav_sweeps_db8_1_10_30_50.npy')
                sw2 = np.load('./Recs/wav_sweeps_db8_2_10_30_50.npy')
            else:
                sw1 = np.load('./Recs/wav_sweeps_db4_1_10_30_50.npy')
                sw2 = np.load('./Recs/wav_sweeps_db4_2_10_30_50.npy')
        else:
            if wav == 'sym8':
                sw1 = np.load('./Recs/wav_sweeps_sym8_filas_1_10_30_50.npy')
                sw2 = np.load('./Recs/wav_sweeps_sym8_filas_2_10_30_50.npy')
            elif wav == 'db8':
                sw1 = np.load('./Recs/wav_sweeps_db8_1_10_30_50_lineas.npy')
                sw2 = np.load('./Recs/wav_sweeps_db8_2_10_30_50_lineas.npy') 
            else:
                sw1 = np.load('./Recs/wav_sweeps_db4_1_10_30_50_lineas.npy')
                sw2 = np.load('./Recs/wav_sweeps_db4_2_10_30_50_lineas.npy')            
        Z = np.zeros([262144,37])
        if s == 0.1:
            Z[:,:32] = sw1[0][:,:]
            Z[:,32:] = sw2[0][:,27:]
        elif s == 0.3:
            Z[:,:32] = sw1[1][:,:]
            Z[:,32:] = sw2[1][:,27:]
        else:
            Z[:,:32] = sw1[2][:,:]
            Z[:,32:] = sw2[2][:,27:]
    else:
        if metodo == 'cs':
            Z = np.load('dct_10_30_50.npy')
        else:
            Z = np.load('dct_10_30_50_filas.npy')
    return Z