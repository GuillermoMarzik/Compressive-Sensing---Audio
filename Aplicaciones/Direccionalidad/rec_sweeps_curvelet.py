#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:40:12 2021

@author: gmarzik
"""

import numpy as np
from rec_curvelet_func import rec_curvelet

Xorig_2 = np.load('sweeps.npy') #Cargo los barridos guardados
gen = Xorig_2[50000:-20000,:] #descarto ceros
del Xorig_2


s = 0.5
x = gen[:,:10] 
#debido al excesivo uso de memoria RAM que no se libera en la librería utilizada para la reconstrucción, se debe resetear el núcleo
#de Python luego de cada tanda de reconstrucciones, cambiar los grupos de barridos que se desea reconstruir y volver a ejecutar, repitiendo
#el proceso hasta que se reconstruyan las 37 señales de interés.  
i = 1
rec_curvelet(x,s,i)

