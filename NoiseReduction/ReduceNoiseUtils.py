#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:06:55 2018

@author: joe
"""
import numpy as np

def computeNormalDirection(normal,magnitud_normal):
    return normal * (1/magnitud_normal)
    
def computeNormalMagnitude(normal):
    return np.linalg.norm(normal) # sqrt(sqr x + sqr y + sqr z)
    
def fixNormalDirection(normal,punto):
    
    suma = 0
    vector_view_punto = -1 * punto #because it's known the viewPoint as (0,0,0)
    
    for pos in range(3):
        suma = suma + (normal[pos] * vector_view_punto[pos])
    
    if suma >= 0:
        return normal
    else:
        return -1*normal

def computeNormal(pc_array,punto_cercano,p1,p2):
    
    pos2 = punto_cercano[p1]
    pos3 = punto_cercano[p2]
    
    p1 = pc_array[punto_cercano[0]]
    p2 = pc_array[pos2]
    p3 = pc_array[pos3]

    p1p2 = p2 - p1
    p1p3 = p3 - p1

    normal = np.cross(p1p2,p1p3)

    return fixNormalDirection(normal, p2)
    

def estimationOfNormals(pc_array,punto_cercano, cantidad):
    
    suma_normal = np.zeros(3)
    cont = 0
    
    for j in range(cantidad):
        #Para no tomar el primer valor que es del mismo punto
        if j > 0:
            if j < cantidad - 1:
                
                normal_intermedia = computeNormal(pc_array,punto_cercano,
                                                    j,j+1)
                
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

def directionOfNormals(pc,kdtree):

    pcArray = pc.to_array()
    
    normalIndex = [] 
    normales = []
    
    quantity = 10
    
    #En la posicion de cada punto de la nube de punto
    for pos in range(pc.size):
        
        #Guardar la posicion del punto
        normalIndex.append(pos)
            
        #Calcular puntos adyacentes
        nearPoint, d = kdtree.nearest_k_search_for_point(pc,pos,
                                                         quantity)
        
        normal = estimationOfNormals(pcArray,nearPoint,quantity)
        #normal = estimationOfMoreRelatedNormals(pcArray,nearPoint,quantity)

        normalMagnitude = computeNormalMagnitude(normal)

        if normalMagnitude != 0:
            
            normalDirection = computeNormalDirection(normal,normalMagnitude)
           
            normales.append(normalDirection)
        else:
            
            normales.append(np.zeros(3))

 
    normalsArray = np.asarray(normales)
    return normalsArray, normalIndex

def directionOfMoreRelatedNormalsMedition(pc,kdtree,quantity):

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

def directionOfNormalsMedition(pc,kdtree,quantity):

    pcArray = pc.to_array()
    
    normalIndex = [] 
    normales = []
    
    #En la posicion de cada punto de la nube de punto
    for pos in range(pc.size):
      
        #Guardar la posicion del punto
        normalIndex.append(pos)
            
        #Calcular puntos adyacentes
        nearPoint, d = kdtree.nearest_k_search_for_point(pc,pos,
                                                         quantity+2)
        
        normal = estimationOfNormals(pcArray,nearPoint,quantity)
      
        #normal = estimationOfMoreRelatedNormals(pcArray,nearPoint,quantity)
        
        normalMagnitude = computeNormalMagnitude(normal)
      
        if normalMagnitude != 0:
            
            normalDirection = computeNormalDirection(normal,normalMagnitude)
            #print (normal[2],normalMagnitude,' -> ',normalDirection)
            normales.append(normalDirection)
        else:
            
            normales.append(np.zeros(3))

    normalsArray = np.asarray(normales)
    return normalsArray, normalIndex
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