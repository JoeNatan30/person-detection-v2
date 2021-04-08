#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:06:55 2018

@author: joe
"""
import numpy as np
import math
import open3d as o3d


def computeNormalDirection(normal, magnitud_normal):
    return normal * (1/magnitud_normal)


def computeNormalMagnitude(normal):
    # sqrt(sqr x + sqr y + sqr z)
    return np.linalg.norm(normal)


def fixNormalDirection(normal, punto):

    # point => because it's known the viewPoint as (0,0,0)
    vector_view_punto = -1 * punto

    result = np.dot(vector_view_punto, normal)

    if result >= 0:
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

    '''
    for j in range(cantidad):
        #Para no tomar el primer valor que es del mismo punto
        if j > 0:
            
            if j < cantidad - 1:
                
               for k in range(cantidad):
                   
                if k + j + 1 >= cantidad: break #en caso salga del rango
               
                k_temp = k + j + 1
                
                normal_intermedia = computeNormal(pc_array,punto_cercano,
                                                    j,k_temp)
                
                suma_normal += normal_intermedia
                
                cont += 1


    En caso se necesite obtener una relacion completa entre todos los puntos
    for j in range(cantidad):
        
        if j > 0:
            
            for k in range(cantidad):
                
                if k + j + 1 >= cantidad: break #en caso salga del rango
               
                k_temp = k + j + 1
                
                normal_intermedia = calcular_normal(pc_array,punto_cercano,
                                                    j,k_temp)
           
                suma_normal = suma_normal + normal_intermedia
                

                cont = cont + 1
    '''

    normal = suma_normal/cont

    return normal


def getNormalDirection(pcd, kdtree, cant):

    pc_array = np.asarray(pcd.points)
    
    pointArr = []
    normalsArr = []

    # En la posicion de cada punto de la nube de punto
    for pos, point in enumerate(pc_array):

        if(pos % 10000 == 0):
            print(pos)

        # Si no es una coordenada vacia
        if not math.isnan(point[0]):

            # Calcular puntos adyacentes
            _, nearPoint, d = kdtree.search_knn_vector_3d(point, cant)

            normal = estimationOfNormals(pc_array, nearPoint, cant)

            normalMagnitude = computeNormalMagnitude(normal)

            if normalMagnitude != 0:

                normalDir = computeNormalDirection(normal, normalMagnitude)
                normalsArr.append(normalDir)
                pointArr.append(point)
            else:
                # para evitar tener valores [0 0 0] como normal
                # esto es para evitar divisiones entre cero mas adelante

                normal[0] = 0.0000000000001
                normal[1] = 0.0000000000001
                normal[2] = 0.0000000000001
                normalsArr.append(normal)
                pointArr.append(point)

    return np.asarray(pointArr), np.asarray(normalsArr)
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
