#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:43:55 2019

@author: guillem
"""
import numpy as np
#import sys
import math

def filtropromediomovil(magnitud,frecuencia,octava):
    fsup = np.zeros(len(frecuencia))
    finf = np.zeros(len(frecuencia))
    fsup[:] = 2**(1/(2*octava))*frecuencia
    finf[:] = frecuencia/(2**(1/(2*octava)))
    a = np.ravel(np.array(np.where(finf < frecuencia[0])))
    indicesup = np.zeros(len(frecuencia))
    indiceinf = np.zeros(len(frecuencia))
    b = np.ravel(np.array(np.where(fsup > frecuencia[-1])));
    lenb = len(b)
    c = np.arange(lenb)
    lenfsup = len(fsup);
    lenf = len(frecuencia);
    for i in a:
        finf[i] = frecuencia[0]   
    for i in c:
        fsup[b[i]] = frecuencia[-1]   
    for i in range(lenfsup):
        minimadif = np.min(np.abs(frecuencia-fsup[i]))
        indicesup[i] = np.array(np.where(np.abs(frecuencia-fsup[i]) == minimadif))
    
    for i in range(lenfsup):
        minimadif = np.min(np.abs(frecuencia-finf[i]))
        indiceinf[i] = np.array(np.where(np.abs(frecuencia-finf[i]) == minimadif))
    prom = np.zeros(lenf)
    largo = np.arange(lenf)
    for i in largo:
        prom[i] = np.sum(10**(magnitud[np.int(indiceinf[i]):np.int(indicesup[i])]/20))
        prom[i] = prom[i] / ((np.int(indicesup[i])-np.int(indiceinf[i]))+1)
        prom[i] = 20*np.log10(prom[i])
        
    return prom
#    prom = np.zeros(len(frecuencia))
#    largo = np.arange(len(frecuencia))
#    for i in largo:
#        aux4 = np.array(np.where(aux[i] == aux1[i:]))
#        indicesup[1,i] = aux4[1,0]
#        aux5 = np.array(np.where(aux3[i] == aux2[i:]))
#        indiceinf[1,i] = aux5[1,0] 
#        prom[i] = np.sum(10**(magnitud[np.int(indiceinf[1,i]):np.int(indicesup[1,i])]))
#        prom[i] = prom[i]/((np.int(indicesup[1,i])-np.int(indiceinf[1,i]))+1)
#        prom[i] = prom[i]+sys.float_info.min
#        prom[i] = 10*math.log10(prom[i])
#    prom = prom /10
#    prom = prom - prom[329] 
#    return prom   
#    return indicesup,indiceinf   
#    fsup = fsup[:,None]
#    aux1 = np.absolute(np.subtract(fsup,frecuencia))
#    aux1 = np.absolute(fsup-frecuencia)
#    aux = np.amin(aux1,axis=1)
#    indicesup = np.zeros((2,len(frecuencia)))
#    finf = finf[:,None]
#    aux2 = np.absolute(np.subtract(finf,frecuencia))
#    aux3 = np.amin(aux2,axis=1)
#    indiceinf = np.zeros((2,len(frecuencia)))
#    prom = np.zeros(len(frecuencia))
#    largo = np.arange(len(frecuencia))
#    for i in largo:
#        aux4 = np.array(np.where(aux[i] == aux1[i:]))
#        indicesup[1,i] = aux4[1,0]
#        aux5 = np.array(np.where(aux3[i] == aux2[i:]))
#        indiceinf[1,i] = aux5[1,0] 
#        prom[i] = np.sum(10**(magnitud[np.int(indiceinf[1,i]):np.int(indicesup[1,i])]))
#        prom[i] = prom[i]/((np.int(indicesup[1,i])-np.int(indiceinf[1,i]))+1)
#        prom[i] = prom[i]+sys.float_info.min
#        prom[i] = 10*math.log10(prom[i])
#    prom = prom /10
#    prom = prom - prom[329] 
#    return prom
