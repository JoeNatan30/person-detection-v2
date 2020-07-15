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

    #To avoid distance elimiation problem (this step its not important, just to reduce data)
    if (len(nearPointIndex) < 0.1* pc.size):
        print ("Here is the Problem <<-------||<")
        return pc
    else:
        return newPc

###############################################################################
# Eliminacion de ruido por normales
###############################################################################
    
def isInAngle(dir_1,dir_2,rangeOfDiff):
    
    cosineOfAngle = np.dot(dir_1,dir_2)
    
    #Because some of this values are close to 1 or -1
    if(cosineOfAngle>1):
        cosineOfAngle=1
    elif (cosineOfAngle<-1):
        cosineOfAngle=-1
    #####
    
    angle = math.acos(cosineOfAngle)
    
    if angle < rangeOfDiff:
        return True
    else:
        return False

def getNotSimilarNormals(normalDirection ,rangeOfDiff ,pos):
    
    direction_1 = normalDirection[pos]
    direction_2 = normalDirection[pos+1]

    ########################################
    #
    #         dir_1    dir_2
    #             ^    ^
    #             |   /
    #             |ö /
    # se salva    |-/  Se salva
    #     ________|/______________
    #   
    #       ö = angle between dir_1 and dir2
    #
    ########################################}
    
    if (not isInAngle(direction_1,direction_2,rangeOfDiff)):
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
    
    #Se genera un nuevo Point cloud sin planos
    if (verbose): print ("removeSimilarPointsUsingNormals")
    pcWithoutFlatPart= removeSimilarPointsUsingNormals(
            pc,
            normalDirection,
            normalIndex,
            rangeOfDiff)
    
    if (verbose): print ("KdtreeStructure.getKdtreeFromPointCloud")
    kdtreeWithoutFlatPart = KdtreeStructure.getKdtreeFromPointCloud(pcWithoutFlatPart)
    
    #Se limpia de los outliears
    cleansedPc = reduceDistancePoint(pcWithoutFlatPart, kdtreeWithoutFlatPart,verbose)
    
    #Nuevo kdtree sin ruido
    cleansedkdtree = KdtreeStructure.getKdtreeFromPointCloud(cleansedPc)
    
    return cleansedPc, cleansedkdtree

###############################################################################
#Aquí va las direcciones de lectura y escritura
    
def ruido(rangeOfDiff, pos, verbose):

    readDir = './../datos/inicial/inicial_%d.pcd'% pos
    writeDir = './../datos/sin_ruido/sin_ruido_%d.pcd'% pos
    
    #Lectura
    if (verbose): print ("READ - ruido")
    pc,kdtree = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    print('Tamaño inicial: ',pc.size)
    
    #Proceso
    if (verbose): print ("PROCESS - ruido")
    cleansedPc, cleansedKdtree = init(pc,kdtree,rangeOfDiff,verbose)
    
    #Escritura
    if (verbose): print ("WRITE - ruido")
    cleansedPc.to_file(str.encode(writeDir))
    
    return cleansedPc, cleansedKdtree
###############################################################################
    
def medition(rangeOfDiff, pos, tipo, precision, verbose):
    
    readDir = './../datos/inicial/inicial_%d.pcd'% pos
    #writeDir = './sin_ruido_%d.pcd'% pos
    
    #Lectura
    if (verbose): print ("READ")
    pc,kdtree = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    
    if(tipo == "ruido-rangeOfDiff"):
        initRSDifftMedition(pc,kdtree,pos, verbose)
    
    if(tipo == "ruido-rangeOfSamples"):
        initRSSamplesMedition(pc,kdtree,rangeOfDiff, verbose)
    
    if(tipo == "ruido-normalPrecision"):
        initNormalPrecision(pc,kdtree,rangeOfDiff,precision,verbose)
    #Escritura
    #if (verbose): print ("WRITE")
    #cleansedPc.to_file(str.encode(writeDir))
    
    #return cleansedPc, cleansedKdtree

def initRSDifftMedition(pc,kdtree,pos, verbose):

    print("Total size: ", pc.size)
    normalDirection, normalIndex = ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition(pc,kdtree,5)
    #normalDirection, normalIndex = ReduceNoiseUtils.directionOfNormalsMedition(pc,kdtree,50)
    
    print(normalDirection)
    
    #rangeStart = 0.00000000149
    rangeStart = 0.000000014889
    rangeLong  = 0.000000000001
    rangeNumber = 1500
    rangeList = []
    
    print (rangeStart)
    print ("")
    for segmentPos in range (rangeNumber):
        rangeList.append(rangeStart + rangeLong * segmentPos)
    '''
    rangeStart = 1
    rangenumber = 100
    rangeLong = 100000
    rangeList = []
    
    for segmentPos in range (rangenumber):
        rangeList.append(rangeStart / rangeLong*segmentPos)
     
    '''
    contadorRango = 0
    for rangeOfDiff in rangeList:
        
        pcWithoutFlatPart = removeSimilarPointsUsingNormals(
                pc,
                normalDirection,
                normalIndex,
                rangeOfDiff)
        
        print (contadorRango,'=> ',(pcWithoutFlatPart.size*100/pc.size))
        contadorRango +=1
    
def initRSSamplesMedition(pc,kdtree,rangeOfDiff, verbose):
    

    print ('Total',',',pc.size)
    print ('MAX',',','Tiempo normal simple',',','Tiempo remove normal simple',',','Número de puntos simple',',','Tiempo normal complex',',','Tiempo remove normal complex',',','Número de puntos complex')
    for cant in range(10):
        
        if(cant== 0 or cant == 1 or cant == 2): continue
        
        #Obtener las normales  y sus indices
        if (verbose): print ("ReduceNoiseUtils.directionOfNormalsMedition")
        start_time = time.time()
        normalDirection1, normalIndex = ReduceNoiseUtils.directionOfNormalsMedition(pc,kdtree,cant + 1)
        print(normalDirection1)
        simple = (time.time() - start_time)
        
        if (verbose): print ("removeSimilarPointsUsingNormals")
        start_time = time.time()
        pcWithoutFlatPart1 = removeSimilarPointsUsingNormals(
                pc,
                normalDirection1,
                normalIndex,
                rangeOfDiff)
        
        simple2 = (time.time() - start_time)
        
        numPointSimple = pcWithoutFlatPart1.size
        
        #######################################################################
        if (verbose): print ("ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition")
        start_time = time.time()
        normalDirection2, normalIndex = ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition(pc,kdtree, cant + 1)
        print(normalDirection2)
        comple = (time.time() - start_time)
        
        if (verbose): print ("removeSimilarPointsUsingNormals")
        start_time = time.time()
        pcWithoutFlatPart2 = removeSimilarPointsUsingNormals(
                pc,
                normalDirection2,
                normalIndex,
                rangeOfDiff)
        
        comple2 = (time.time() - start_time)
        
        numPointcomplex = pcWithoutFlatPart2.size
        
        print (cant+1,',',simple,',',simple2,',',numPointSimple,',',comple,',',comple2,',',numPointcomplex)

def initNormalPrecision(pc,kdtree,rangeOfDiff,precision,verbose):
   
    number = 2000
    
    pc_array = pc.to_array()
    
    pointRange = int(pc.size / number)
    
    pointArr = []
    normalArr = []
    
    for pointTimes in range(number - 2): # -2 to avoid overflow pc size
        
        pointTaked = pointRange * (pointTimes+1) # +1 to avoid take cero value
        
        pointArr.append(pointTaked)
    
    pc_2 = pc.extract(pointArr)
    
    nearPoints, d = kdtree.nearest_k_search_for_cloud(pc_2,precision+1) # +1 because kdtree will return the point itself
    
    for oneNearPoint in nearPoints:
        
        normal = ReduceNoiseUtils.estimationOfMoreRelatedNormals(pc_array,oneNearPoint,precision+1) # +1 to avoid use the point itself
        normalMagnitude = ReduceNoiseUtils.computeNormalMagnitude(normal)
        
        if normalMagnitude != 0:
            
            normalDirection = ReduceNoiseUtils.computeNormalDirection(normal,normalMagnitude)
            normalArr.append(normalDirection)
        else:
            
            normalArr.append(np.zeros(3))
            
    for num in range(len(nearPoints)):       
        print(num,',',normalArr[num][0],',',normalArr[num][1],',',normalArr[num][2],',')
    
    # get normals from these maps and save the precition
