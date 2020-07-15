# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:07:27 2019

@author: Joe
"""

import numpy as np

def isInTheRange(valor,minimo,maximo):
    
    if valor >= minimo and valor <= maximo:
        return True
    else:
        return False

def erraseUsedPointInArray(size, usedPointIndex):
    
    notUsedInd = []
    
    for val in np.arange(size):
        
        if not(val in usedPointIndex):
            notUsedInd.append(val)
    
    return np.array(notUsedInd)

def createNewFPFH(fpfh, new_list_ind):
    
    fpfh_list = []
    
    for pos in new_list_ind:
        fpfh_list.append(fpfh[pos])
    
    return np.array(fpfh_list)

def getExtremValues(valor,minimo,maximo):
    
    if (valor < minimo):
        minimo = valor
        
    if (valor > maximo):
        maximo = valor
        
    return minimo, maximo

def operations(x,y,z):

    #Posicion real de los puntos x,y,z obtenidos de Skeleton
    pc_x = ((x -160.0) * z) / 150.0
    pc_y = ((y - 120.0) * z) / 165.0
    pc_z = z**2.15 / 6300.0
    
    #Transformacion matricial
    extrinsic = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[0., 0., 1., 0.],
                          [0., 0., 0., 1.]], np.float64)
    extrinsic
    inv_extr = np.linalg.inv(extrinsic)
    punto = np.array([pc_x,pc_y,pc_z,1.0])

    #Porcentaje de reduccion general de los puntos
    matriz = inv_extr * punto / 100.0
    
    #Para hacer girar la imagen
    transform = np.array([[1., 0., 0., 0.],[0., -1., 0., 0.],[0., 0., -1., 0.],
                          [0., 0., 0., 1.]], np.float64)
    matriz = matriz * transform
    vector_matriz = np.array([matriz[0][0],matriz[1][1],matriz[2][2]],dtype = np.float32)
     
    return vector_matriz 

def transformation(x,y,z):
    
    pos_punto = np.array([x, y, z, 1.0])
    
    transform = np.array([[1., 0., 0., 0.],[0., -1., 0., 0.],[0., 0., -1., 0.],
                          [0., 0., 0., 1.]], np.float64)
    
    matriz = pos_punto * transform
    
    return np.array([matriz[0][0],matriz[1][1],matriz[2][2]])
