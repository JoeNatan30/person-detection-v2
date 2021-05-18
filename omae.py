# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 18:17:12 2021

@author: Joe
"""
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import open3d as o3d
from NoiseReduction import KdtreeStructure
from Characterization import Fpfh
from random import randint

def name():
    dtp = pd.read_pickle("./../datos/dataset/dataTrainX.pkl")
    dataset_x = dtp['dataTrainX']

    print(len(dataset_x[0])/3200)
###############################################################################


def showTricontour(results, x, y, xlim, ylim, nameX, nameY, modelName):

    fig, ax = plt.subplots(1, 1)

    ax.tricontour(x, y, results, levels=14,
                  linewidths=0.5, colors='k')
    cntr = ax.tricontourf(x, y, results,
                          levels=14, cmap="RdBu_r")

    fig.colorbar(cntr, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_title("Estimador F1 (%s)" % modelName)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
###############################################################################


def randomForestInfo():

    data = []

    for line in open('./results/rf_ovr_f1_final.json', 'r'):
        data.append(json.loads(line))

    results = []
    n_estimators = []
    max_features = []
    min_samples_split = []

    for rslt in data:

        for params in rslt['params'].items():
            if(params[0] == 'n_estimators'):
                x = params[1]
                n_estimators.append(params[1])
                neL = (50, 1000)

            if(params[0] == 'min_samples_split'):
                z = params[1]
                min_samples_split.append(params[1])
                mssL = (2, 150)

            if(params[0] == 'max_features'):
                y = params[1]
                #max_features.append(1/(pow(10, params[1])))
                max_features.append(params[1])
                mfL = (0, 3)

        results.append(rslt['target'])

#    showTricontour(results, n_estimators, max_features, neL, mfL, "Estimadores", "Características máximas")
    showTricontour(results, n_estimators, min_samples_split,
                   neL, mssL,
                   "N° de estimadores", "División mímima de la muestra",
                   "Bosque aleatorios")
#    showTricontour(results, min_samples_split, max_features, mssL, mfL, "División mímima de la muestra", "Características máximas")


def SVMInfo():

    data = []

    for line in open('./results/f1_macro/svc_f1_macro.json', 'r'):
        data.append(json.loads(line))

    results = []
    expC = []
    expGamma = []

    for rslt in data:

        for params in rslt['params'].items():
            if(params[0] == 'expC'):
                x = 2 ** params[1]
                expC.append(params[1])
                ecL = (-5, 30)

            if(params[0] == 'expGamma'):
                y = 2 ** params[1]
                expGamma.append(params[1])
                egL = (-15, 4)

        results.append(rslt['target'])

    showTricontour(results, expC, expGamma,
                   ecL, egL, 
                   "exp C", "exp Gamma", 
                   "SVM")


def NeuralNetworkInfo():
    
    data = []

    for line in open('./results/f1_macro/nn_f1_macro_log.json', 'r'):
        data.append(json.loads(line))

    results = []
    hidden_layer_sizes_1 = []
    hidden_layer_sizes_2 = []
    for rslt in data:

        for params in rslt['params'].items():

            if(params[0] == 'hidden_layer_sizes_1'):
                d = params[1]
                hidden_layer_sizes_1.append(params[1])
                hls1L = (20, 716)

            if(params[0] == 'hidden_layer_sizes_2'):
                e = params[1]
                hidden_layer_sizes_2.append(params[1])
                hls2L = (20, 715)

        results.append(rslt['target'])

    showTricontour(results, hidden_layer_sizes_1, hidden_layer_sizes_2,
                   hls1L, hls2L,
                   "N° de neuronas escondidas 1", "N° de neuronas escondidas 2",
                   "Redes neuronales")

def errasePointInPcd(points, posList):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.select_by_index(posList, invert=True)

def getKdtreeFromPointCloud(pcd):

    return o3d.geometry.KDTreeFlann(pcd)

def showPoints(pcd):
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.1,
                                      front=[0, 0, 1],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0])

def histograma(pc, kdtree, tamano, verbose):

    if(verbose):
        print("Fpfh.inicio")
    list_fpfh_point = Fpfh.inicio(pc, kdtree, tamano, verbose)

    return list_fpfh_point

def PruebaReal():
    
    pc_seg, kdtree_seg, pcdSize = KdtreeStructure.getKdtreeFromPointCloudDir(
        './../segmentado_2680.pcd')
    
    pcArrRetrieved = []
    fpfh_list_retrieved = []
    
    segmentSize = 800
    
    repetitions = int(pcdSize/segmentSize)
    print("## %d ##" % repetitions)
    for _ in range(repetitions):
        print(_)
        #Actual structure with data
        pcArr = np.asarray(pc_seg.points)
        kdtree_seg = getKdtreeFromPointCloud(pc_seg)
        
        # get nearPoints
        pos = randint(0, len(pcArr - 1))
        point = pcArr[pos,:]

        _, nearPoint, d = kdtree_seg.search_knn_vector_3d(point, segmentSize)

        # results
        fpfh_list = histograma(pc_seg, kdtree_seg, segmentSize, False)

        # preparing new structure and data
        pc_seg = errasePointInPcd(pc_seg.points, nearPoint)

        #Retrieved info
        pcArrRetrieved.append([pcArr[val] for val in nearPoint])
        fpfh_list_retrieved.append(fpfh_list[pos])

    #print(pcArrRetrieved)
    #print(np.asarray(fpfh_list_retrieved))
    #print([len(val) for val in fpfh_list_retrieved])
    
    modelo_filename = './Results/nn_final.pkl'
    mpl = joblib.load(modelo_filename)
    y_pred = mpl.predict(np.asarray(fpfh_list_retrieved))
    
    print(y_pred)

    color = [[255, 0, 0],[255, 255, 0], [0, 255, 0], [0, 0, 255]]

    newPcArr = []
    pcColor = []
    for ind, pointList in enumerate(pcArrRetrieved):
        for point in pointList:
            newPcArr.append(point)
            pcColor.append(color[y_pred[ind]])
    
    newPcArr = np.asarray(newPcArr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(newPcArr)
    pcd.colors = o3d.utility.Vector3dVector(pcColor)

    showPoints(pcd)
    
PruebaReal()
#SVMInfo()
#NeuralNetworkInfo()
#randomForestInfo()
