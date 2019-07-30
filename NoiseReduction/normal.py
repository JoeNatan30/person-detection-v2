#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:06:55 2018

@author: joe
"""
import numpy as np
import math

def calcular_direccion_normal(normal,magnitud_normal):
    return normal * (1/magnitud_normal)
    
def calcular_magnitud_normal(normal):
    return (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5

def calcular_normal(pc_array,punto_cercano):
    
    p1 = pc_array[punto_cercano[0]]
    p2 = pc_array[punto_cercano[1]]
    p3 = pc_array[punto_cercano[2]]
    
    p1p2 = p2 - p1
    p1p3 = p3 - p1
    
    return np.cross(p1p2,p1p3)

def calcular_estimacion_normal(pc_array,punto_cercano, cantidad):
    
    suma_normal = np.zeros(3)
    cont = 0
    
    for j in range(cantidad):
        #Para no tomar el primer valor que es del mismo punto
        if j > 0:
            if j < cantidad - 1:
                
                normal_intermedia = calcular_normal(pc_array,punto_cercano,
                                                    j,j+1)
        
                suma_normal += normal_intermedia
                
                cont += 1
    
    normal = suma_normal/cont

    return normal
    
def obtener_direcciones_normales(pc,kdtree):

    pc_array = pc.to_array()
    
    normal_indices = [] 
    normales = []
    
    #En la posicion de cada punto de la nube de punto
    for pos in range(pc.size):
        
        #Si no es una coordenada vacia
        if not math.isnan(pc_array[pos][0]):
            
            #Guardar la posicion del punto
            normal_indices.append(pos)
            
            #Calcular puntos adyacentes
            punto_cercano, distancia = kdtree.nearest_k_search_for_point(pc,pos,7)

            normal = calcular_normal(pc_array,punto_cercano)
            #normal = calcular_estimacion_normal(pc_array,punto_cercano,6)
            
            magnitud_normal = calcular_magnitud_normal(normal)

            if magnitud_normal != 0:
                
                direccion_normal = calcular_direccion_normal(normal,magnitud_normal)
                normales.append(direccion_normal)
            else:
                #En caso que la normal valga cero
                normales.append(np.zeros(3))

    normales_array = np.asarray(normales)
    return normales_array, normal_indices



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
