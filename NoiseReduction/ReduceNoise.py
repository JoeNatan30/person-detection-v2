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

import pandas as pd
import matplotlib.pyplot as ptl
from mpl_toolkits.mplot3d import Axes3D


###############################################################################
# Eliminacion ruido por distancias
###############################################################################
def reduceDistancePoint(pcd, kdtree, v):

    pointsArr = ReduceNoiseUtils.getArrFromPcd(pcd.points)
    normalsArr = ReduceNoiseUtils.getArrFromPcd(pcd.normals)

    if v == 1:
        rango = 0.7
        num_punt = 5
    else:
        rango = 0.005
        num_punt = 5

    newPoints = []
    newNormals = []
    '''
    The idea of this "For" is to save all out of range index points
    and then extract these from the point cloud
    '''
    for pos, point in enumerate(pointsArr):

        # Si no es una coordenada vacia
        if not math.isnan(point[0]):

            # Get near points
            _, nearPoint, d = kdtree.search_knn_vector_3d(point, num_punt*2)

            cant = 0

            for dist in d:
                # if this point is inside the range
                if dist < rango:
                    cant = cant + 1

            # To filter isolated points
            if cant > num_punt:
                newPoints.append(point)
                newNormals.append(normalsArr[pos])

    newPcd = ReduceNoiseUtils.getPcdFromPointsAndNormals(pointsArr, normalsArr)

    return newPcd


###############################################################################
# Eliminacion de ruido por normales
###############################################################################
def isInAngle(dir_1, dir_2, rangeOfDiff):

    cosineOfAngle = np.dot(dir_1, dir_2)

    # Because some of this values are close to 1 or -1
    if(cosineOfAngle > 1):
        cosineOfAngle = 1
    elif (cosineOfAngle < -1):
        cosineOfAngle = -1
    #####

    angle = math.acos(cosineOfAngle)

    if angle < rangeOfDiff:
        return True
    else:
        return False


def getNotSimilarNormals(normalDirection, rangeOfDiff, pos):

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
    ########################################

    if (not isInAngle(direction_1, direction_2, rangeOfDiff)):
        return True
    else:
        return False


def removeSimilarPointsUsingNormals(pcd, rangeOfDiff):

    pointszArr = ReduceNoiseUtils.getArrFromPcd(pcd.points)
    normalArr = ReduceNoiseUtils.getArrFromPcd(pcd.normals)

    takenPoints = []
    takenNormals = []

    for pos, normal in enumerate(normalArr):

        # Si son normales distintas entonces se guarda (se verifica que pos+1
        # no sea mayor al tamaño del arreglo de normales)
        if (len(normalArr) > pos+1) and getNotSimilarNormals(normalArr,
                                                             rangeOfDiff, pos):
            takenNormals.append(normal)
            takenPoints.append(pointszArr[pos])

    newPcd = ReduceNoiseUtils.getPcdFromPointsAndNormals(takenPoints,
                                                         takenNormals)
    return newPcd


def init(pcd, kdtree, rangeOfDiff, verbose):

    # Obtener las normales  y sus indices
    if(verbose):
        print("ReduceNoiseUtils.directionOfNormals")
    # ReduceNoiseUtils.showPointCloud(pcd)
    pcd = ReduceNoiseUtils.directionOfNormals(pcd, kdtree)

    # Se genera un nuevo Point cloud sin planos
    if (verbose):
        print("removeSimilarPointsUsingNormals")

    pcdWithoutFlatPart = removeSimilarPointsUsingNormals(pcd, rangeOfDiff)

    if (verbose):
        print("KdtreeStructure.getKdtreeFromPointCloud")
    kdtreeWithoutFlatPart = KdtreeStructure.getKdtreeFromPointCloud(
        pcdWithoutFlatPart)

    # Se limpia de los outliears
    cleansedPcd = reduceDistancePoint(pcdWithoutFlatPart,
                                      kdtreeWithoutFlatPart, verbose)
    # ReduceNoiseUtils.showPointCloud(cleansedPcd)
    print("Tamaño reducido a un: %2.2f" %(100*len(np.asarray(cleansedPcd.points))/len(np.asarray(pcd.points))))
    # Nuevo kdtree sin ruido
    cleansedkdtree = KdtreeStructure.getKdtreeFromPointCloud(cleansedPcd)

    return cleansedPcd, cleansedkdtree


###############################################################################
# Aquí va las direcciones de lectura y escritura
def ruido(rangeOfDiff, pos, verbose):

    readDir = './../datos/inicial/inicial_%d.pcd' % pos
    writeDir = './../datos/sin_ruido/sin_ruido_%d.pcd' % pos
    # Lectura
    if(verbose):
        print("READ - ruido")

    pcd, kdtree, size = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)
    print('Tamaño inicial: ', size)
    # ReduceNoiseUtils.showPointCloud(pcd)
    # Proceso
    if (verbose):
        print("PROCESS - ruido")
    cleansedPcd, cleansedKdtree = init(pcd, kdtree, rangeOfDiff, verbose)

    # Escritura
    if (verbose):
        print("WRITE - ruido")
    # ReduceNoiseUtils.showPointCloud(cleansedPcd)
    # TODO verify if it save correctly
    ReduceNoiseUtils.saveFile(writeDir, cleansedPcd)

    return cleansedPcd, cleansedKdtree


##############################################################################
def medition(rangeOfDiff, pos, tipo, precision, verbose):

    readDir = './../datos/inicial/inicial_%d.pcd' % pos
    #writeDir = './sin_ruido_%d.pcd' % pos

    #Lectura
    if (verbose): print ("READ")
    pcd, kdtree, size = KdtreeStructure.getKdtreeFromPointCloudDir(readDir)

    if(tipo == "ruido-rangeOfDiff"):
        initRSDifftMedition(pcd, kdtree, verbose)
        #initRSDiffCantMedition(pc, kdtree, verbose)

    if(tipo == "ruido-rangeOfSamples"):
        initRSSamplesMedition(pcd, kdtree, rangeOfDiff, verbose)

    if(tipo == "ruido-normalPrecision"):
        initNormalPrecision(pcd, kdtree, rangeOfDiff, precision, verbose)

    if(tipo == "plot-reduction-rangeDiffVsPointsAcum"):
        initPlotReductionRdVsPa(pcd, kdtree, verbose)
    #Escritura
    #if (verbose): print ("WRITE")
    #cleansedPc.to_file(str.encode(writeDir))

    #return cleansedPc, cleansedKdtree


def initPlotReductionRdVsPa(pcd, kdtree, verbose):

    size = len(np.asarray(pcd.points))
    print("Total size: ", size)

    rangeArr_x = np.linspace(0.001, 0.1, 100)
    points_y = np.linspace(10, 150, 15)

    results = np.zeros((len(rangeArr_x), len(points_y)))

    for pos_y, point in enumerate(points_y):

        pcdReduced = ReduceNoiseUtils.directionOfNormalsMedition(pcd, kdtree,
                                                                 int(point))
        for pos_x, diff in enumerate(rangeArr_x):
            pcdReduced = removeSimilarPointsUsingNormals(pcd, diff)
            results[pos_x][pos_y] = (1 - (len(np.asarray(pcdReduced.points))
                                          / size))*100
            print("[%3d][%3.3f] ==> %3.2f" % (point, diff, results[pos_x][pos_y]))

    X, Y = np.meshgrid(points_y, rangeArr_x)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, results, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('número de puntos')
    ax.set_ylabel('rango de diferencia')

    ax.set_zlabel('Porcentaje reducido')
    ax.set_title('Porcentaje de reducción de ruido')

    fig2, ax2 = plt.subplots(1, 1)
    contourf_ = ax2.contourf(X, Y, results, cmap='viridis')

    ax2.set_title('Porcentaje de reducción de ruido')
    ax2.set_xlabel('número de puntos')
    ax2.set_ylabel('rango de diferencia')
    fig2.colorbar(contourf_)
    plt.show()


def initRSDifftMedition(pcd, kdtree, verbose):

    size = len(np.asarray(pcd.points))
    print("Total size: ", size)
    #normalDirection, normalIndex = ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition(pc,kdtree,5)
    pcd = ReduceNoiseUtils.directionOfNormalsMedition(pcd, kdtree, 120)

    size = len(np.asarray(pcd.points))
    print(pcd.normals)
    ReduceNoiseUtils.showNormals(pcd)
    #rangeStart = 0.00000000149

    rangeLong = 0.005
    rangeNumber = 100
    rangeList = []

    print("")
    for segmentPos in range(rangeNumber):
        rangeList.append(rangeLong * (segmentPos+1))
    '''
    rangeStart = 1
    rangenumber = 100
    rangeLong = 100000
    rangeList = []

    for segmentPos in range (rangenumber):
        rangeList.append(rangeStart / rangeLong*segmentPos)

    '''

    percent = []

    contadorRango = 0
    for rangeOfDiff in rangeList:

        pcdReduced = removeSimilarPointsUsingNormals(pcd, rangeOfDiff)
        print(contadorRango)
        #print(len(np.asarray(pcdReduced.points)), " while ->   ", size, rangeOfDiff)
        #print(contadorRango, '=> ', "%3.2f" % (len(np.asarray(pcdReduced.points)) *
        #                             100 / size))
        percent.append((1 - len(np.asarray(pcdReduced.points)) / size) * 100)
        contadorRango += 1

    data = {'rango': rangeList,
            'porcentaje reducido': percent}

    df = pd.DataFrame(data, columns=['porcentaje reducido', 'rango'])
    df.plot(x='porcentaje reducido', y='rango', kind='line')
    plt.show()


def initRSDiffCantMedition(pcd, kdtree, verbose):

    size = len(np.asarray(pcd.points))

    print("Total size: ", size)
    #normalDirection, normalIndex = ReduceNoiseUtils.directionOfMoreRelatedNormalsMedition(pc,kdtree,5)

    ReduceNoiseUtils.showNormals(pcd)
    #rangeStart = 0.00000000149
    diff = 0.1

    percent = []
    numbers = []

    for cant in range(5,200):

        pcd = ReduceNoiseUtils.directionOfNormalsMedition(pcd, kdtree, cant)
        
        pcdReduced = removeSimilarPointsUsingNormals(pcd, diff)
        print(cant)
        #print(len(np.asarray(pcdReduced.points)), " while ->   ", size, rangeOfDiff)
        #print(contadorRango, '=> ', "%3.2f" % (len(np.asarray(pcdReduced.points)) *
        #                             100 / size))
        percent.append((1 - len(np.asarray(pcdReduced.points)) / size) * 100)
        numbers.append(cant)

    data = {'rango': numbers,
            'porcentaje reducido': percent}

    df = pd.DataFrame(data, columns=['porcentaje reducido', 'rango'])
    df.plot(x='porcentaje reducido', y='rango', kind='line')
    plt.show()


def initRSSamplesMedition(pc, kdtree, rangeOfDiff, verbose):

    print ('Total',',',pc.size)
    print ('MAX',',','Tiempo normal simple',',','Tiempo remove normal simple',',','Número de puntos simple',',','Tiempo normal complex',',','Tiempo remove normal complex',',','Número de puntos complex')
    for cant in range(10):

        if(cant == 0 or cant == 1 or cant == 2):
            continue

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
