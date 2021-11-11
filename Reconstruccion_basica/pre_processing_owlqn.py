#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:20:46 2021

@author: gmarzik
"""

import numpy as np
import mdct
from scipy.fft import dct, idct
import soundfile as sf
from pylbfgs import owlqn
import pickle
from scipy.stats import bernoulli,poisson
from stdct import sdct,isdct
import math 

def pre_pros_pad(signal,frame_size,window='boxcar',transform='stdct'):
    """
    Esta función toma una señal y la concatena con un vector de ceros para darle el formato requerido por el callback del método OWL-QN

    Parameters
    ----------
    signal : array
        Señal a la que se le van a agregar ceros.
    frame_size : int
        Largo del frame de la transformada tiempo-frecuencia. 
    window : string, opcional
        Tipo de ventana utilizada en los distintos frames. El default es 'boxcar', ventana rectangular.
    transform : string, opcional
        Transformada tiempo-frecuencia elegida como base de representación. El default es 'stdct'.

    Returns
    -------
    signal_pad : array
        Señal formateada para el método OWL-QN.

    """
    frame_step = frame_size.copy()
    if ((transform == 'stdct') or (transform == 'stdht')):
        signal_t = sdct(signal,frame_size,frame_step,window=window)
    elif (transform=='mdct'):
        signal_t = mdct.fast.mdct(signal)
    else:
        signal_t = np.zeros(math.ceil(math.log(len(signal),2)))
    tot_size = signal_t.size
    signal_pad = np.pad(signal,(0,np.int(tot_size-len(signal))))
    return signal_pad    


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

def rec_owlqn(ri,b,c,N,progreso=False):
    """
    Reconstrucción de una señal mediante Compressive Sensing utilizando el algoritmo OWL-QN.
    Basado y adaptado del código de XXX
    REFERENCIA

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
        """An in-memory evaluation callback."""

        # we want to return two things: 
            # (1) the norm squared of the residuals, sum((Ax-b).^2), and
            # (2) the gradient 2*A'(Ax-b)

        # expand x columns-first
        x2 = x.reshape((N)).T

        # Ax is just the inverse 2D dct of x2
        Ax2 = idct(x2, norm='ortho')    

        # stack columns and extract samples
        Ax = Ax2.T.flat[ri].reshape(b.shape)

        # calculate the residual Ax-b and its 2-norm squared
        Axb = Ax - b
        fx = np.sum(Axb ** 2)


        # project residual vector (k x 1) onto blank image (ny x nx)
        Axb2 = np.zeros(Ax2.shape)
        Axb2[ri] = Axb

        # A'(Ax-b) is just the 2D dct of Axb2
        AtAxb2 = 2 * dct(Axb2,norm='ortho')
        AtAxb = AtAxb2.reshape(x.shape)

        # copy over the gradient vector
        np.copyto(g, AtAxb)

        return fx
    def progress(x, g, fx, xnorm, gnorm, step, k, ls):
        return 0
    if progreso:
        Xat2 = owlqn(N, evaluate, None, c)
    else:
        Xat2 = owlqn(N, evaluate, progress, c)
    Xat = Xat2.reshape(N).T
    Xa = idct(Xat,norm='ortho')
    return Xa


