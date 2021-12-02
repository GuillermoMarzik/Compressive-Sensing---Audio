#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:05:17 2021

@author: gmarzik
"""
import numpy as np
import soundfile as sf
from pre_processing_owlqn import subsamp,rec_owlqn,pre_pros_pad

#A signal sampled at Nyquist rate is loaded.
Xorig,fs = sf.read('tono.wav')
#Different percentages of the total number of samples are made available, as well as the corresponding Lambda parameters for use in the Poisson distribution.
sample_sizes = (0.05, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95)
lambda_ = np.array([0.051,0.105,0.17,0.228,0.285,0.36,0.425,0.52,0.6,0.71,0.8,0.91,1.05,1.25,1.35,1.7,1.95,2.4,2.9])

#Padding with zeros (for MDCT, STDCT, STDHT and DWT)
Xorig,dim2 = pre_pros_pad(Xorig,transform='sdct')

#Random samples of the signal are retrieved, following a given probability distribution.
ri,b = subsamp(sample_sizes[10],Xorig,lambda_[10],med_mat = 'Gauss')

transform = 'sdct'
if transform == 'mdct':
    N = len(Xorig) +512
else:
    N = len(Xorig)
c = 0.0001 #orthanwise constant for OWL-QN algorithm, sparsity control
#Signal reconstruction from a given percentage of the total number of samples
Xa = rec_owlqn(ri,b,c,N,progreso=True, transformada='dwt',wav='db4',dim2=dim2)



