#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:30:38 2021

@author: gmarzik
"""

import numpy as np

def FHT_FFT(x):
    """
    Esta función toma un vector x y lo transforma a frecuencia a través de la Transformada de Hartley. Esta se computa de forma rápida aprovechando la relación que existe con la transformada de Fourier cuando el vector a transformar es real.
    
    Parameters
    ----------
    x : array de floats
        Vector a transformar según la FHT.

    Returns
    -------
    x_f_H: array de floats
        Vector transformado.
    """
    x_f = np.fft.fft(x)
    x_f_H = np.real(x_f) - np.imag(x_f)
    return(x_f_H)