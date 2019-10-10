#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:32:51 2018

@author: joe
"""
import math
import numpy as np
import time

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
    
def isOutOfRange(dir_1, dir_2, rangeOfDiff):
    
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
        if value > rangeOfDiff: # 0.4
            return True
           
    #if any of vector components are not out the range
    return False

def getNotSimilarNormals(direccion_normal ,rangeOfDiff ,pos):
    
    direction_1 = direccion_normal[pos]
    direction_2 = direccion_normal[pos+1]
    
    #if both normals are out of range
    if isOutOfRange(direction_1, direction_2, rangeOfDiff):
        return True
    #if a normal with the oposite direction normal are out of range
    if isOutOfRange(direction_1, direction_2 * -1, rangeOfDiff):
        return True
    else:
        return False


def removeSimilarPointsUsingNormals(pc,normalDirection,normalIndex, rangeOfDiff):
    
    takenPoints = []
    
    for pos in range(len(normalIndex)-1):
        
        #Si son normales distintas entonces se guarda
        if getNotSimilarNormals(normalDirection, rangeOfDiff, pos):
             takenPoints.append(normalIndex[pos])

    newPc = pc.extract(takenPoints)

    return newPc


def init(pc,kdtree,rangeOfDiff, verbose):
    
    #Obtener las normales  y sus indices
    if (verbose): print ("ReduceNoiseUtils.directionOfNormals")
    normalDirection, normalIndex = ReduceNoiseUtils.directionOfNormals(pc,kdtree)
    
    #El nuevo Point cloud sin planos
    if (verbose): print ("removeSimilarPointsUsingNormals")
    pcWithoutFlatPart= removeSimilarPointsUsingNormals(
            pc,
            normalDirection,
            normalIndex,
            rangeOfDiff)
    
    if (verbose): print ("KdtreeStructure.getKdtreeFromPointCloud")
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
    
def ruido(rangeOfDiff, pos, verbose):

    readDir = '../inicial/inicial_%d.pcd'% pos
    writeDir = './sin_ruido_%d.pcd'% pos
    
    #Lectura
    if (verbose): print ("READ")
    pc,kdtree = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    
    #Proceso
    if (verbose): print ("PROCESS")
    cleansedPc, cleansedKdtree = init(pc,kdtree,rangeOfDiff,verbose)
    
    #Escritura
    if (verbose): print ("WRITE")
    cleansedPc.to_file(str.encode(writeDir))
    
    return cleansedPc, cleansedKdtree
###############################################################################
    
def medition(rangeOfDiff, pos, verbose):
    
    readDir = '../inicial/inicial_%d.pcd'% pos
    #writeDir = './sin_ruido_%d.pcd'% pos
    
    #Lectura
    if (verbose): print ("READ")
    pc,kdtree = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    
    #Proceso
    if (verbose): print ("MEDITION - INIT")
    cleansedPc, cleansedKdtree = initMedition(pc,kdtree,rangeOfDiff,verbose)
    
    #Escritura
    #if (verbose): print ("WRITE")
    #cleansedPc.to_file(str.encode(writeDir))
    
    #return cleansedPc, cleansedKdtree
    
def initMedition(pc,kdtree,rangeOfDiff, verbose):
    

    print ('Total',',',pc.size)
    for cant in range(10):
       
        #Obtener las normales  y sus indices
        if (verbose): print ("ReduceNoiseUtils.directionOfNormalsMedition")
        start_time = time.time()
        normalDirection1, normalIndex = ReduceNoiseUtils.directionOfNormalsMedition(pc,kdtree,cant + 1)
        simple = (time.time() - start_time)
        
        if (verbose): print ("ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition")
        start_time = time.time()
        normalDirection2, normalIndex = ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition(pc,kdtree, cant + 1)
        comple = (time.time() - start_time)
        
        print (cant+1,',',simple,',',normalDirection1.size,',',comple,',',normalDirection2.size)
