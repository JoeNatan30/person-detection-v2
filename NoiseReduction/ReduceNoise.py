#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:32:51 2018

@author: joe
"""
import pcl
import math
import numpy as np

from NoiseReduction import ReduceNoiseUtils
from NoiseReduction import KdtreeStructure

###############################################################################
# Eliminacion ruido por distancias
###############################################################################
def reduceDistancePoint(pc,kdtree,v):
    
    pc_array = pc.to_array()
    
    if v==1:
        rango = 0.7
        num_punt = 5
    else:
        rango = 0.005
        num_punt = 5
   
    nearPointIndex = []
    
    '''
    The idea of this "For" is to save all out of range index points
    and then extract these from the point cloud
    '''
    for pos in range(pc.size):
        
        #Si no es una coordenada vacia
        if not math.isnan(pc_array[pos][0]):
            
            #Get near points
            nearPoint, distance = kdtree.nearest_k_search_for_point(pc,pos,
                                                                    num_punt*2)
            cant = 0
            
            for dist in distance:
                #if this point is inside the range
                if dist < rango:
                    cant = cant + 1    
            
            #To save out of range points
            if cant > num_punt:   
                nearPointIndex.append(pos)
    
    #This step is necesary to be able to use pc.extract
    nearPointIndexArray = np.asarray(nearPointIndex)

    newPc = pc.extract(nearPointIndexArray)

    return newPc

###############################################################################
# Eliminacion de ruido por normales
###############################################################################
    
def isOutOfRange(dir_1,dir_2):
    
    dirVariation = []
    
    for pos in range(3):
        dirVariation.append(math.fabs(dir_1[pos]-dir_2[pos]))
 
    """
    The idea of this For is to get almost one component that is out of the 
    chosen range to say the function is true
    """
    for value in dirVariation:
        
        #return true if its diffece is more than the chosen value
        #if the chosen value is reduced,
        #it's possible to get more points
        #
        #         dir_1    dir_2
        #             ^    ^
        #             | X /
        #             |  /
        # se salva    | /  Se salva
        #     ________|/______________
        #   
        #       x = dir_1 - dir_2
        #
        #if value > 0.003:
        if value > 0.4:
            return True
           
    #if any of vector components are not out the range
    return False

def getNotSimilarNormals(direccion_normal,pos):
    
    direction_1 = direccion_normal[pos]
    direction_2 = direccion_normal[pos+1]
    
    #if both normals are out of range
    if isOutOfRange(direction_1,direction_2):
        return True
    #if a normal with the oposite direction normal are out of range
    if isOutOfRange(direction_1,direction_2 * -1):
        return True
    else:
        return False


def removeSimilarPointsUsingNormals(pc,normalDirection,normalIndex):
    
    takenPoints = []
    
    for pos in range(len(normalIndex)-1):
        
        #Si son normales distintas entonces se guarda
        if getNotSimilarNormals(normalDirection,pos):
             takenPoints.append(normalIndex[pos])

    newPc = pc.extract(takenPoints)

    return newPc


def init(pc,kdtree):
    
    #Obtener las normales  y sus indices
    normalDirection, normalIndex = ReduceNoiseUtils.directionOfNormals(pc,kdtree)
    
    #El nuevo Point cloud sin planos
    pcWithoutFlatPart= removeSimilarPointsUsingNormals(pc,normalDirection,normalIndex)
    
    kdtreeWithoutFlatPart = KdtreeStructure.getKdtreeFromPointCloud(pcWithoutFlatPart)

    
    '''
    #Se limpia de los outliears
    pc_sin_ruido = sin_puntos_lejanos_distancias(pc_sin_planos,
                                                 kdtree_sin_planos)
    #pc_sin_ruido.to_file('data/sin_ruido/sin_plano/sin_plano4.pcd')
    
    #Nuevo kdtree sin ruido
    kdtree_sin_ruido = KdtreeStructure.obtencion_kdtree_from_pointCloud(pc_sin_ruido)
    '''
    return pcWithoutFlatPart, kdtreeWithoutFlatPart

###############################################################################
#Aquí va las direcciones de lectura y escritura
    
def ruido(pos):

    readDir = '../inicial/inicial_%d.pcd'% pos
    writeDir = './sin_ruido_%d.pcd'% pos
    
    #Lectura
    pc,kdtree = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    
    #Proceso
    cleansedPc, cleansedKdtree = init(pc,kdtree)
    
    #Escritura
    cleansedPc.to_file(str.encode(writeDir))
    
    return cleansedPc, cleansedKdtree
###############################################################################