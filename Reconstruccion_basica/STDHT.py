#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:45:09 2021

@author: gmarzik
"""

import numpy as np
from FHT import FHT_FFT

def stdht(len_signal,nperseg):
    dht_mxm = FHT_FFT(np.eye(nperseg))
    dht_ext = np.kron(np.eye(np.int(len_signal/nperseg)),dht_mxm)
    idht_ext = np.kron(np.eye(np.int(len_signal/nperseg)),np.linalg.inv(dht_mxm))
    return dht_ext,idht_ext