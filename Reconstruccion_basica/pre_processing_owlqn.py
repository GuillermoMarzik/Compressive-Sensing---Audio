#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:20:46 2021

@author: gmarzik
"""

import numpy as np
import mdct
from scipy.fft import dct, idct
from pylbfgs import owlqn
from scipy.stats import bernoulli,poisson
from stdct import sdct,isdct
import math 
from FHT import FHT_FFT
from stdht import sdht,isdht
import pywt

def pre_pros_pad(signal,transform='stdct'):
    """
    Esta función toma una señal y la concatena con un vector de ceros para darle el formato requerido por el callback del método OWL-QN

    Parameters
    ----------
    signal : array
        Señal a la que se le van a agregar ceros.
    transform : string, opcional
        Transformada tiempo-frecuencia elegida como base de representación. El default es 'stdct'.

    Returns
    -------
    signal_pad : array
        Señal formateada para el método OWL-QN.

    """
    dim2 = None
    if ((transform == 'stdct') or (transform == 'stdht') or (transform=='mdct')):
        signal_t = mdct.fast.mdct(signal)
        N_aux = signal_t.shape[0]*signal_t.shape[1]
        dim2 = signal_t.shape[1]
        tot_size = N_aux-512
    else:
        #wavelet
        signal_t = math.ceil(math.log(len(signal),2))
        tot_size = 2**signal_t
    signal_pad = np.pad(signal,(0,np.int(tot_size-len(signal))))
    if transform == 'sdct' or transform == 'stht':
        signal_t = sdct(signal_pad,2048,2048,window='boxcar')
        dim2 = signal_t.shape[1]
    return signal_pad,dim2


def subsamp(s, signal, lambda_, med_mat = 'Gauss'):
    """
    Selección aleatoria de elementos de un vector mediante alguna distribución de probabilidad. Primera etapa del método Compressive Sensing.

    Parameters
    ----------
    s : float
        Fracción de las muestras de la señal que se toman para el proceso.
    signal : float
        Array con la señal sobre la cual se aplica el proceso.
    lambda_ : float
        Parámetro lambda de la distribución Poisson.
    med_mat : string, optional
        Distribución de probabilidad elegida para la selección de muestras. El default es 'Gauss'.

    Returns
    -------
    ri : int
        Array de índices seleccionados para la reconstrucción.
    b: float
        Array de valores de las muestras en los índices ri.

    """
    N = signal.shape
    if len(N)==1:
        N = N[0]
        if med_mat == 'Gauss':
            k = round(N * s)
            ri = np.random.choice(N, k, replace=False) 
        elif med_mat == 'Bernoulli':
            k = bernoulli.rvs(s,size =N)
            ri = np.where(k == 1)[0]
        elif med_mat == 'Poisson':
            k = poisson.rvs(lambda_, size = N)
            ri = np.where(k != 0)[0]
        else:
            print('No es una matriz válida')
            return
    else:
        nx,ny = N
        if med_mat == 'Gauss':
            k = round(nx* ny * s)
            ri = np.random.choice(nx*ny, k, replace=False) 
        elif med_mat == 'Bernoulli':
            k = bernoulli.rvs(s,size =nx*ny)
            ri = np.where(k == 1)[0]
        elif med_mat == 'Poisson':
            k = poisson.rvs(lambda_, size = nx*ny)
            ri = np.where(k != 0)[0]
        else:
            print('No es una matriz válida')
            return        
    b = signal.T.flat[ri].astype(float)
    return ri,b   

def rec_owlqn(ri,b,c,N,progreso=False,transformada='dct',wav = None,dim2=None):
    """
    Reconstrucción de una señal mediante Compressive Sensing utilizando el algoritmo OWL-QN.
    Basado y adaptado del código de Robert Taylor
    http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

    Parameters
    ----------
    ri : int
        Array de índices de las muestras utilizadas para el proceso.
    b : float
        Array de valores de las muestras en los índices ri.
    c : float
        Orthanwise constant para el método OWL-QN.
    N : int
        Cantidad de muestras totales de la señal a reconstruir.
    progreso : bool
        Mostrar o no por consola el progreso del algoritmo. Por defecto False.

    Returns
    -------
    Xa: float
        Array con la señal reconstruida.

    """

    def evaluate(x, g, step):
        if transformada == 'dct':
            x2 = x.reshape((N)).T
            Ax2 = idct(x2, norm='ortho')
        elif transformada == 'sdct':
            x2 = x.reshape((2048,dim2))
            Ax2 = isdct(x2,frame_step=2048,frame_length=2048,window='boxcar')
        elif transformada == 'stht':
            x2 = x.reshape((2048,dim2))
            Ax2 = isdht(x2,frame_step=2048,frame_length=2048,window='boxcar')       
        elif transformada == 'dht':
            x2 = x.reshape((N)).T
            Ax2 = FHT_FFT(x2)
        elif transformada == 'mdct':
            x2 = x.reshape((512,dim2))
            Ax2 = mdct.fast.imdct(x2)
        elif transformada == 'dwt':
            x2 = x.reshape((N)).T
            x3 = pywt.array_to_coeffs(x2,coff_shape,output_format='wavedec')
            Ax2 = pywt.waverec(x3, wav,mode='periodization')

        Ax = Ax2.T.flat[ri].reshape(b.shape)

        Axb = Ax - b
        fx = np.sum(Axb ** 2)

        if transformada == 'dct' or transformada == 'sdct' or transformada == 'sdht' or transformada == 'dht' or transformada == 'dwt':
            Axb2 = np.zeros(Ax2.shape)
            Axb2[ri] = Axb
        elif transformada == 'mdct':
            Axb2 = np.zeros((x.shape)[0]-512)
            Axb2.T.flat[ri] = Axb
            
        if transformada == 'dct':
            AtAxb2 = 2 * dct(Axb2,norm='ortho')
        elif transformada == 'sdct':
            AtAxb2 = 2 * sdct(Axb2,2048,2048,window='boxcar')
        elif transformada == 'stht':
            AtAxb2 = 2 * sdht(Axb2,2048,2048,window='boxcar')      
        elif transformada == 'dht':
            AtAxb2 = 2 * FHT_FFT(Axb2)
        elif transformada == 'mdct':
            AtAxb2 = 2 * mdct.fast.mdct(Axb2)
        elif transformada == 'dwt':
            AtAxb2 = pywt.wavedec(Axb2,wav,mode='periodization')
            AtAxb2 = 2 * pywt.coeffs_to_array(AtAxb2)[0]

        AtAxb = AtAxb2.reshape(x.shape)

        np.copyto(g, AtAxb)

        return fx
    def progress(x, g, fx, xnorm, gnorm, step, k, ls):
        return 0
    if transformada == 'dwt':
        aux = np.zeros(N)
        aux_w = pywt.wavedec(aux,wav,mode='periodization')
        global coff_shape
        _,coff_shape= pywt.coeffs_to_array(aux_w)
        del aux, aux_w
    if progreso:
        Xat2 = owlqn(N, evaluate, None, c)
    else:
        Xat2 = owlqn(N, evaluate, progress, c)

    if transformada == 'sdct':
        Xat = Xat2.reshape((2048,dim2))
        Xa = isdct(Xat,frame_step=2048,frame_length=2048,window='boxcar')
    if transformada == 'sdht':
        Xat = Xat2.reshape((2048,dim2))
        Xa = isdht(Xat,frame_step=2048,frame_length=2048,window='boxcar')        
    elif transformada == 'mdct':
        Xat = Xat2.reshape((512,dim2))
        Xa = mdct.fast.imdct(Xat)
    elif transformada == 'dwt':
        Xat2 = Xat2.ravel()
        Xat = pywt.array_to_coeffs(Xat2,coff_shape,output_format='wavedec')
        Xa = pywt.waverec(Xat,wav,mode='periodization')
    elif transformada == 'dct':    
        Xat = Xat2.reshape(N).T
        Xa = idct(Xat,norm='ortho')
    elif transformada == 'dht':
        Xat = Xat2.reshape(N).T
        Xa = FHT_FFT(Xat)       
    return Xa


