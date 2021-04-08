#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:06:55 2018

@author: joe
"""
import numpy as np
import open3d as o3d


def computeNormalDirection(normal, magnitud_normal):
    return normal * (1/magnitud_normal)


def computeNormalMagnitude(normal):
    # sqrt(sqr x + sqr y + sqr z)
    return np.linalg.norm(normal)


def fixNormalDirection(normal, punto):

    suma = 0
    # Punto is used because it's known the viewPoint as (0,0,0)
    vector_view_punto = -1 * punto

    for pos in range(3):
        suma = suma + (normal[pos] * vector_view_punto[pos])

    if suma >= 0:
        return normal
    else:
        return -1*normal


def computeNormal(pc_array, punto_cercano, p1, p2):

    pos2 = punto_cercano[p1]
    pos3 = punto_cercano[p2]

    p1 = pc_array[punto_cercano[0]]
    p2 = pc_array[pos2]
    p3 = pc_array[pos3]

    p1p2 = p2 - p1
    p1p3 = p3 - p1

    normal = np.cross(p1p2, p1p3)

    return fixNormalDirection(normal, p2)


def estimationOfNormals(pc_array, punto_cercano, cantidad):

    suma_normal = np.zeros(3)
    cont = 0

    for j in range(cantidad):
        # Para no tomar el primer valor que es del mismo punto
        if j > 0:

            if j < cantidad - 1:

                normal_intermedia = computeNormal(pc_array, punto_cercano,
                                                  j, j+1)

                suma_normal += normal_intermedia

                cont = cont + 1

    normal = suma_normal/cont

    return normal

def estimationOfMoreRelatedNormals(pc_array,punto_cercano, cantidad):

    suma_normal = np.zeros(3)
    cont = 0

    for j in range(cantidad):
        #Para no tomar el primer valor que es del mismo punto
        if j > 0:

            if j < cantidad - 1:

               for k in range(cantidad):

                if k + j + 1 >= cantidad: continue #en caso salga del rango

                k_temp = k + j + 1

                normal_intermedia = computeNormal(pc_array,punto_cercano,
                                                    j,k_temp)

                suma_normal += normal_intermedia

                cont += 1

    normal = suma_normal/cont

    return normal


def directionOfNormals(pcd, kdtree):

    pcdArr = np.asarray(pcd.points)
    quantity = 100
    '''
    normals = []

    

    # En la posicion de cada punto de la nube de punto
    for pos, point in enumerate(pcdArr):

        # Calcular puntos adyacentes
        _, nearPoint, d = kdtree.search_knn_vector_3d(point, quantity)

        normal = estimationOfNormals(pcdArr, nearPoint, quantity)
        #normal = estimationOfMoreRelatedNormals(pcArray, nearPoint, quantity)

        normalMagnitude = computeNormalMagnitude(normal)

        if normalMagnitude != 0:

            normalDirection = computeNormalDirection(normal, normalMagnitude)

            normals.append(normalDirection)
        else:

            normals.append(np.zeros(3))

    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    '''
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=5, max_nn=quantity))

    normalsarr = []

    for pos, normal in enumerate(pcd.normals):
        newNormal = fixNormalDirection(normal, pcdArr[pos])
        normalMagnitude = computeNormalMagnitude(newNormal)

        if normalMagnitude != 0:
            normalDirection = computeNormalDirection(newNormal,
                                                     normalMagnitude)

            normalsarr.append(normalDirection)
        else:
            normalsarr.append(np.zeros(3))

    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normalsarr))


    return pcd


def directionOfMoreRelatedNormalsMedition(pc, kdtree, quantity):

    pcArray = pc.to_array()

    normalIndex = []
    normales = []

    #En la posicion de cada punto de la nube de punto
    for pos in range(pc.size):

        #Guardar la posicion del punto
        normalIndex.append(pos)

        #Calcular puntos adyacentes
        nearPoint, d = kdtree.nearest_k_search_for_point(pc,pos,
                                                         quantity)

        #normal = estimationOfNormals(pcArray,nearPoint,quantity)
        normal = estimationOfMoreRelatedNormals(pcArray,nearPoint,quantity)

        normalMagnitude = computeNormalMagnitude(normal)

        if normalMagnitude != 0:

            normalDirection = computeNormalDirection(normal,normalMagnitude)
            normales.append(normalDirection)
        else:

            normales.append(np.zeros(3))

    normalsArray = np.asarray(normales)
    return normalsArray, normalIndex


def directionOfNormalsMedition(pcd, kdtree, quantity):

    pcdArr = np.asarray(pcd.points)
    '''
    normals = []

    # En la posicion de cada punto de la nube de punto
    for pos, point in enumerate(pcdArr):

        # Calcular puntos adyacentes
        _, nearPoint, d = kdtree.search_knn_vector_3d(point, quantity + 1)

        normal = estimationOfNormals(pcdArr, nearPoint, quantity)
        #normal = estimationOfMoreRelatedNormals(pcArray, nearPoint, quantity)

        normalMagnitude = computeNormalMagnitude(normal)

        if normalMagnitude != 0:

            normalDirection = computeNormalDirection(normal, normalMagnitude)

            normals.append(normalDirection)
        else:

            normals.append(np.zeros(3))

    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    '''
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=5, max_nn=quantity))

    normalsarr = []

    for pos, normal in enumerate(pcd.normals):
        newNormal = fixNormalDirection(normal, pcdArr[pos])
        normalMagnitude = computeNormalMagnitude(newNormal)

        if normalMagnitude != 0:
            normalDirection = computeNormalDirection(newNormal,
                                                     normalMagnitude)

            normalsarr.append(normalDirection)
        else:
            normalsarr.append(np.zeros(3))

    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normalsarr))

    

    return pcd


def getArrFromPcd(pcd):
    return np.asarray(pcd)


def getPcdFromPointsAndNormals(points, normals):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def saveFile(writeDir, cleansedPcd):
    o3d.io.write_point_cloud(writeDir, cleansedPcd)


def readFile(readDir, form):

    return o3d.io.read_point_cloud(readDir, format=form)


def showNormals(pcd):

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.1,
                                      front=[0, 0, 1],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0],
                                      point_show_normal=True)


def showPointCloud(pcd):

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.1,
                                      front=[0, 0, 1],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0])
'''
def obtener_normal_arreglo_kdtree(arrPc):

    pc_normal = []

    for pc in arrPc:
        pc_normal.append(obtener_normal_kdtree(pc))

    return pc_normal

def obtener_normal_escena():

    pc = []
    pc = estructura.obtencion_kdtree_escena(10)

    return obtener_normal_arreglo_kdtree(pc)
'''
