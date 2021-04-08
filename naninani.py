# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:48:22 2021

@author: Joe
"""
import pandas as pd
import numpy as np
import open3d as o3d

from NoiseReduction import ReduceNoise, KdtreeStructure, ReduceNoiseUtils
from Segmentation import RansacUtils
from Segmentation import RansacAlgorithm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential(x, a, b):
    #return a*np.exp(b*x)
    return a + b * np.log(x)
    #return a/(x+b)

def tryOne():
    
    posList =[6, 9, 16, 18, 23, 29, 35, 37, 38, 40, 43, 46, 54, 55, 56, 57, 58, 60, 67, 68, 69, 70, 74, 75, 76, 83, 94, 98, 103, 116, 118, 119, 120, 130, 157, 159, 160, 170, 172, 173, 174, 178, 179, 180, 187, 188, 190, 192, 193, 194, 195, 207, 208, 209, 210, 212, 216, 218, 219, 220, 232, 233, 234, 235, 245, 246, 247, 257, 258, 260, 261, 265, 266, 267, 268, 344, 347]
    posList = [58]
    '''
    for pos in posList:
        print("POS===: ", pos)
        pcd = KdtreeStructure.getPointCloudFromDir('./../datos/inicial/inicial_%d.pcd' % pos)

        RansacUtils.showPoints(pcd)
    '''
    
    for pos in posList:
        print("POS===: ", pos)
        pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                        0.02, int(pos), True)
        
        pc_arr = RansacUtils.getArrFromPcd(pc_sin_ruido.points)
        
        plane_arr = RansacAlgorithm.getPlaneSectionPcd(pc_arr, 0.8)
        total = len(plane_arr)
        newPcd = RansacUtils.getPcdFromPoints(plane_arr)
        RansacUtils.showPoints(newPcd)
        
        rango = 0.3 + 0.03*0
        part_arr = RansacAlgorithm.getPlaneSectionPcd(plane_arr, rango)
        print("rango: ", rango)
        part = len(part_arr)
        
        print(part/total)


def lastPlot():
    data = pd.read_json("./medition_ransac_results_manual_last.json")
    dataMod = pd.read_json("./medition_ransac_results_manual_last_mod.json")

    dataS = data.sort_index(level=0)
    
    for mod in dataMod:
        dataS[mod].update(dataMod[mod])
        
    for val in dataS:
        dataS[val].update(dataS[val]*100)

    dataT = dataS.T
    dataT.columns = [float("{:4.2f}".format(x)) for x in np.linspace(0.2, 1.7, 31)]

    mean = []

    for pos, row in enumerate(dataT):
        mean.append(np.mean(dataT[row]))
    
    ranguito = []
    for pos in range(len(mean)-1):
        ranguito.append(mean[pos+1] - mean[pos])
    ranguito.append(ranguito[len(ranguito)-1])
    meanArr = np.asarray(mean)
    meanArr = meanArr * 100
    '''
    plt.figure(figsize=(7, 5))
    plt.scatter(np.linspace(0.2, 1.7, 31), meanArr, s=64)
    plt.title("Gráfico de dispersión")
    plt.ylabel("porcentaje reducido (%)")
    plt.xlabel("distancia punto a recta")

    pars, cov = curve_fit(f=exponential, xdata=np.linspace(0.2, 1.7, 31),
                          ydata=meanArr, p0=[0, 0], bounds=(-np.inf, np.inf))

    a = pars[0]
    b = pars[1]

    #print(np.log(1/(a*b))/b)
    print(b)
    print(a, b)

    '''
    plt.figure(figsize=(8, 6))
    boxplot = dataT.boxplot()
    title_boxplot = 'Diagrama de cajas'
    plt.title( title_boxplot )
    plt.xticks(rotation=45)
    plt.xlabel("distancia punto a recta")
    plt.ylabel("porcentaje del plano reducido (%)")
    plt.suptitle('')  # that's what you're after
    plt.show()

def plotear():

    data = pd.read_json("./medition_ransac_results.json")
    dataMod = pd.read_json("./medition_ransac_results_mod.json")
    dataLast = pd.read_json("./medition_ransac_results_mod_2.json")
    dataNig = pd.read_json("./medition_ransac_results_mod_3.json")

    dataS = data.sort_index(level=0)

    #for mod in dataMod:
    #    dataS[mod].update(dataMod[mod])

    for last in dataLast:
        dataS[last].update(dataLast[last])

    for nig in dataNig:
        dataS[nig].update(dataNig[nig])

    for pos, row in enumerate(dataS):
        if(dataS[row][0] < 0.3):
            print(dataS[row])
            #del dataS[row]

    dataT = dataS.T

    dataT.columns = [float("{:4.2f}".format(x)) for x in np.linspace(0.2, 0.8, 20)]
    #boxplot = dataT.boxplot(vert=False)

    # df is your dataframe
    mean = []

    for pos, row in enumerate(dataT):
        mean.append(np.mean(dataT[row]))

    print(mean)
    print( [float("{:4.2f}".format(x)) for x in np.linspace(0.2, 0.8, 20)])
    plt.plot(mean, color='blue', marker='o', linestyle='dashed', linewidth=1,
             markersize=8)
    plt.title("Grafica de tabla comparativa")
    plt.show()
    
    '''
    title_boxplot = 'Diagrama de cajas'
    plt.title( title_boxplot )
    plt.xlabel("porcentaje del plano reducido")
    plt.ylabel("distancia punto a recta")
    plt.suptitle('')  # that's what you're after
    plt.show()
    dataT.hist2d()
    '''
    
def savingData():
    
    for pos in range(4000):
        
        if(pos>2999):
            continue
        
        pcdSR = KdtreeStructure.getPointCloudFromDir('./../datos/\sin_ruido/\sin_ruido_%d.pcd' % pos)

        pcdSeg = KdtreeStructure.getPointCloudFromDir('./../datos/segmentado/segmentado_%d.pcd' % pos)


        
        dataS = np.asarray(pcdSR.points)
        dataR = np.asarray(pcdSeg.points)
        
        xyz = []
        xyzR = []
        
        for posi, row in enumerate(dataS):
            if(dataS[posi][1] < -17):
                xyz.append(dataS[posi])
                
        for posi, row in enumerate(dataR):
            if(dataR[posi][1] < -17):
                xyzR.append(dataR[posi])
                
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        pcdR = o3d.geometry.PointCloud()
        pcdR.points = o3d.utility.Vector3dVector(xyzR)
        
        percent = (1 - len(pcdR.points)/len(pcd.points))*100

        if(pos < 2999 and percent == 0.0):
            
            xyz = []
            for posi, row in enumerate(dataS):
                if(dataS[posi][1] > -17):
                    xyz.append(dataS[posi])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            #   RansacUtils.showPoints(pcd)
            #kdtreeSR = KdtreeStructure.getKdtreeFromPointCloud(pcdSR)
            #pc_seg, kdtree_sin_seg = RansacAlgorithm.iniciar(pcdSR, kdtreeSR, 1)
            #RansacUtils.showPoints(pcd)
            writeDir = './../datos/segmentado/segmentado_%d.pcd' % pos
            ReduceNoiseUtils.saveFile(writeDir, pcd)

            print(pos, percent)

            
savingData()
#lastPlot()
#plotear()

'''
pcd, kdtree_sin_ruido = ReduceNoise.ruido(
                        0.02, int(1), True)
RansacUtils.showPoints(pcd)


pc_arr = RansacUtils.getArrFromPcd(pcd.points)

new = []

for point in pc_arr:
    if(point[1] < -10):
        new.append(point)

newPcd = RansacUtils.getPcdFromPoints(new)
RansacUtils.showPoints(newPcd)
'''
