#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:53:15 2021

@author: gmarzik
"""

import scipy.fftpack as spfft
import numpy as np
import pywt
from pylbfgs import owlqn


def subsamp(nx,ny,s,CS=True):
    """
    Función para submuestrear una señal de entrada.

    Parameters
    ----------
    inx,ny : int
        Dimensiones de la matriz a reconstruir.
    s : float
        Proporción de muestras a conservar (1 = todas, 0.5 = mitad, etc).
    CS : bool
        Método de sensado compresivo puro o método híbrido. Default, sensado compresivo puro.

    Returns
    -------
    ri : array
        Índices de las muestras conservadas.

    """
    if CS:
        k = round(nx * ny * s)
        ri = np.random.choice(nx * ny, k, replace=False)
    else:
        k = round(nx*s)
        ri = np.random.choice(nx, k, replace=False)
        ri_t = ri*ny
        aux = []
        for l in range(np.max(ri)+1):
            if(len(np.where(ri_t==l*ny)[0]) != 0):
                a = np.array(range(l*ny+1,(l+1)*ny))
                aux.append(a)
        aux = np.array(aux)
        aux_x,aux_y = aux.shape
        aux = aux.reshape(aux_x*aux_y)
        ri = np.concatenate([ri_t,aux])
    return ri 

def cs_rec(ri,b,nx,ny,c,transform='dct',wav='db4'):
    """
    Función para reconstruir instancias del cuerpo de datos.

    Parameters
    ----------
    ri : array
        Índices de las muestras conservadas.
    b : array
        Valor de las muestras conservadas.
    nx : int
        Cantidad de muestras en sentido horizontal de cada instancia.
    ny : int
        Cantidad de muestras en sentido vertical de cada instancia.
    c : float
        Orthanwise constant, control de densidad de la transformada de la instancia.
    transform : string, optional
        Transformada utilizada como base de representación. El default es 'dct', la otra opción es 'dwt'.
    wav : string, optional
        Familia de wavelets utilizada para la DWT. El default es 'db4'.

    Returns
    -------
    Xa1 : ndarray
        Instancia reconstruida.

    """
    def dct2(x):
        return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

    def idct2(x):
        return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
    
    def evaluate(x, g, step):
        x2 = x.reshape((nx, ny)).T
        
        if transform == 'dwt':
            x3 = pywt.array_to_coeffs(x2,coff_shape,output_format='wavedec2')
            Ax2 = pywt.waverec2(x3,wav,mode='periodization')
        else:
            Ax2 = idct2(x2)
        

        Ax = Ax2.T.flat[ri].reshape(b.shape)

        Axb = Ax - b
        fx = np.sum(np.power(Axb, 2))

        Axb2 = np.zeros(x2.shape)
        Axb2.T.flat[ri] = Axb 
        if transform == 'dwt':
            AtAxb2 = pywt.wavedec2(Axb2,wav,mode='periodization')
            AtAxb2 = 2 * pywt.coeffs_to_array(AtAxb2)[0]
        else:
            AtAxb2 = 2 * dct2(Axb2)
        AtAxb = AtAxb2.T.reshape(x.shape) 

        np.copyto(g, AtAxb)
    

        return fx
    def progress(x, g, fx, xnorm, gnorm, step, k, ls):
        return 0

    aux = np.zeros((ny,nx))
    print(nx,ny)
    aux_w = pywt.wavedec2(aux, wav,mode='periodization')
    global coff_shape
    _,coff_shape= pywt.coeffs_to_array(aux_w)
    del aux, aux_w
    Xat2 = owlqn(nx*ny, evaluate, None, c)
    Xat = Xat2.reshape(nx, ny).T
    if transform == 'dwt':
        Xat = pywt.array_to_coeffs(Xat,coff_shape,output_format='wavedec2')
        Xa = pywt.waverec2(Xat,wav,mode='periodization')
    else:
        Xa = idct2(Xat)
    Xa1 = Xa.T.clip(min=0)
    return Xa1