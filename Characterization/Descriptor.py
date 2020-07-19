#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:31:26 2018

@author: joe
"""
import os
import pcl
import numpy as np
import xml.etree.ElementTree as et
from random import randint

from Characterization import DescriptorUtils

def es_parte_persona(val_extremos,punto):

    x_min_per = val_extremos[0]
    y_min_per = val_extremos[1]
    z_min_per = val_extremos[2]
    
    x_max_per = val_extremos[3]
    y_max_per = val_extremos[4]
    z_max_per = val_extremos[5]
    
    x = punto[0]
    y = punto[1]
    z = punto[2]
    
    if DescriptorUtils.isInTheRange(x, x_min_per, x_max_per):
        if DescriptorUtils.isInTheRange(y, y_min_per, y_max_per):
            if DescriptorUtils.isInTheRange(z, z_min_per, z_max_per):
                return True
    
    return False

def categoria_entrenamiento(pc,val_personas_extremos, conjunto_ind):
    
    pc_arr = pc.to_array()
    
    ind_conj_tomado = np.zeros(len(conjunto_ind))
    
    lista_cat_persona = []
    lista_cat_no_persona = []
    
    for pos_persona in range(len(val_personas_extremos)):
        
        cat_persona = []
        
        for pos_conjunto in range(len(conjunto_ind)):
            
            if ind_conj_tomado[pos_conjunto] == 0:
                    
                parte = conjunto_ind[pos_conjunto]               
                ind_pos_punto = randint(0,len(parte)-1)

                pos_punto = parte[ind_pos_punto]
                
                punto = pc_arr[pos_punto]
                
                if es_parte_persona(val_personas_extremos[pos_persona],punto):
                    
                    ind_conj_tomado[pos_conjunto] = 1
                    
                    cat_persona.append(parte)
                    
        lista_cat_persona.append(cat_persona)
    
    for pos_conjunto in range(len(conjunto_ind)):
        
        if (ind_conj_tomado[pos_conjunto] == 0):
            
            lista_cat_no_persona.append(conjunto_ind[pos_conjunto])
    
    return lista_cat_persona, lista_cat_no_persona

def dentro_rango2(pos_min, pos_max, minimo, maximo):
    
    if pos_max >= minimo and pos_min <= maximo:
        return True
    else:
        return False

def es_parte_persona2(val_extremos,extremos):

    x_min_per = val_extremos[0]
    y_min_per = val_extremos[1]
    z_min_per = val_extremos[2]
    
    x_max_per = val_extremos[3]
    y_max_per = val_extremos[4]
    z_max_per = val_extremos[5]
    
    x_min = extremos[0]
    y_min = extremos[1]
    z_min = extremos[2]
    
    x_max = extremos[3]
    y_max = extremos[4]
    z_max = extremos[5]
    
    if dentro_rango2(x_min, x_max, x_min_per, x_max_per):
        if dentro_rango2(y_min, y_max, y_min_per, y_max_per):
            if dentro_rango2(z_min, z_max, z_min_per, z_max_per):
                return True
    
    return False


def categoria_entrenamiento2(val_personas_extremos, parte_extremo, conjunto_ind):
    
    ind_conj_tomado = np.zeros(len(parte_extremo))
    
    lista_cat_persona = []
    lista_cat_no_persona = []
    
    for pos_persona in range(len(val_personas_extremos)):
        
        cat_persona = []
        
        for pos_seccion in range(len(parte_extremo)):
            
            extremos = parte_extremo[pos_seccion]
            
            if ind_conj_tomado[pos_seccion] == 0:
                
                if es_parte_persona2(val_personas_extremos[pos_persona],extremos):
                    
                    ind_conj_tomado[pos_seccion] = 1
                    
                    cat_persona.append(conjunto_ind[pos_seccion])
                    
        lista_cat_persona.append(cat_persona)
    
    for pos_seccion in range(len(parte_extremo)):
        
        if (ind_conj_tomado[pos_seccion] == 0):
            
            lista_cat_no_persona.append(conjunto_ind[pos_seccion])
    
    return lista_cat_persona, lista_cat_no_persona


###############################################################################
def quitar_segmento(list_general, arr_conjunto):
    
    ind_general = np.zeros(len(list_general))
    
    new_general = []
    
    for punto in arr_conjunto:
        ind_general[punto] = 1
    
    for pos in range(len(list_general)):
        if(ind_general[pos] == 0):
            new_general.append(pos)
    
    return new_general

###############################################################################

def leer_xml(pos):
    
    base_path = os.path.dirname(os.path.realpath(__file__))
    
    xml_file = os.path.join(base_path,'./../../datos/skeleton/Skeleton %d.xml'%pos)
    
    tree = et.parse(xml_file)
    
    root = tree.getroot()    
    
    return root

def obtener_descriptores_train(pos,list_fpfh_point, pc, kdtree,cant_puntos):
    
    root = leer_xml(pos)
    print (root)
    print (pc.size)
    partes_selec = {0:"Head", #cabeza = 0
                    1:"ShoulderCenter",2:"Spine", #torso = 1
                    3:"ElbowRight",4:"WristRight",5:"ElbowLeft",6:"WristLeft", #brazos = 2
                    7:"KneeRight",8:"KneeLeft",9:"FootRight",10:"FootLeft"} #pierna = 3
    
    #partes_selec = {0:"Head",   #cabeza = 0
    #                1:"ShoulderCenter",2:"Spine", #torso = 1
    #                3:"ElbowRight",4:"WristRight",5:"ElbowLeft",6:"WristLeft", #brazos = 2
    #                7:"KneeRight",8:"KneeLeft"} #pierna = 3

    cant_partes = len(partes_selec)
    
    pc_arr = pc.to_array() #Point cloud de la data convertido en array
    
    descriptionSet = []
    extremSet = []
    answerSet = []
    
    indice_parte_cuerpo = -1 #Variable usada en conjunto_resp
    # -1 = nulo
    # 0 = cabeza
    # 1 = torso
    # 2 = brazos
    # 3 = piernas
    
    #Por cada persona (esqueleto) en escena
    for skeleton in root:
        
        playerId = int(skeleton.find('PlayerId').text)
        
        if(playerId > 0): #Entra si hay una persona
            
            joints = skeleton.find('Joints')
            
            #por cada parte de la persona
            for parte in joints:
                
                nombre_parte = parte.find('JointType').text
                
                #si esa parte esta en el diccionario
                for n_dic in range(cant_partes):
                    
                    if nombre_parte == partes_selec[n_dic]: 
                        
                        coord = parte.find('PositionToDepth')
                        
                        #Obtener los valores x, y, z (Sin procesar)
                        x = np.float64(coord.find("X").text)
                        y = np.float64(coord.find("Y").text)
                        z = np.float64(coord.find("Depth").text)
                        
                        #Procesar los datos x, y, z
                        coord_arr = DescriptorUtils.operations(x,y,z)
                        
                        #Se convierte en arreglo de dos dimensiones
                        pcl_one_coord = np.array([coord_arr],dtype = 'float32')
                        
                        #Obtener el pc de los partes
                        pc_one_coord = pcl.PointCloud()
                        pc_one_coord.from_array(pcl_one_coord)
                        
                        posicion = 0 #ya que solo tiene un valor pc_coord
                        
                        #Obtener los puntos cercanos del punto obtenido de Skeleton
                        pos_puntos_cercanos_skl, dist = kdtree.nearest_k_search_for_point(pc_one_coord,
                                                                            posicion,
                                                                            cant_puntos)
                        ####
                        #Para equilibrar la data (segun zonas del cuerpo)
                        
                        #cabeza
                        if n_dic == 0: 
                            indice_parte_cuerpo = 0
                            numVariation = 4
                            #Tomar un valor al azar del conjunto de puntos cercanos
                            variationPointInd = [randint(0, cant_puntos - 1) for _ in range(numVariation)]
                            

                        #torso
                        if n_dic == 1 or n_dic == 2: 
                            indice_parte_cuerpo = 1
                            numVariation = 2
                            #Tomar un valor al azar del conjunto de puntos cercanos
                            variationPointInd = [randint(0, cant_puntos - 1) for _ in range(numVariation)]
                        
                        #brazo
                        if n_dic == 3 or n_dic == 4 or n_dic == 5 or n_dic == 6: 
                            indice_parte_cuerpo = 2
                            #No variation
                            variationPointInd = [0] #index of the first point selected
                            
                        #pierna 
                        if n_dic == 7 or n_dic == 8 or n_dic == 9 or n_dic == 10: 
                            indice_parte_cuerpo = 3
                             #No variation
                            variationPointInd = [0] #index of the first point selected
                            
                        for pointInd in variationPointInd:
                            
                            #obtener los puntos cercanos al punto elegido
                            punto_cercano, d = kdtree.nearest_k_search_for_point(pc,
                                                                 pos_puntos_cercanos_skl[pointInd],
                                                                 cant_puntos)
                            
                            
                       
                      
                            #Obtener el conjunto de descriptores y valores extremos del la seccion de puntos
                            descripcion, extrem = getExtremAndDescription(pc_arr,
                                                                          list_fpfh_point,
                                                                          punto_cercano)                            
   
                            descriptionSet.append(descripcion)
                            extremSet.append(extrem)
                            answerSet.append(indice_parte_cuerpo) 
                            
    return descriptionSet, answerSet, extremSet

###############################################################################
def getExtremAndDescription(pc_arr, list_fpfh, punto_cercano):
    
    alpha = []
    phi   = []
    theta = []
    
    x_min =  np.inf
    x_max = -np.inf
    
    y_min =  np.inf
    y_max = -np.inf
    
    z_min =  np.inf
    z_max = -np.inf

    #El indice de cada punto cercano
    for ind in punto_cercano:
        
        punto = pc_arr[ind]
        
        x_min, x_max = DescriptorUtils.getExtremValues(punto[0], x_min, x_max)
        y_min, y_max = DescriptorUtils.getExtremValues(punto[1], y_min, y_max)
        z_min, z_max = DescriptorUtils.getExtremValues(punto[2], z_min, z_max)
        
        alpha.append(list_fpfh[ind][0])
        phi.append(list_fpfh[ind][1])
        theta.append(list_fpfh[ind][2])
    
    #orden x,y,z (min) + x,y,z (max)
    extremValues = [x_min, y_min, z_min, x_max, y_max, z_max]
    
    alpha_hist, alpha_bin_edges = np.histogram(alpha, bins=40, range=(0.0, 4.0), density=True)
    phi_hist,   phi_bin_edges   = np.histogram(phi,   bins=40, range=(0.0, 4.0), density=True)
    theta_hist, theta_bin_edges = np.histogram(theta, bins=40, range=(0.0, 4.0), density=True)
    
    descripcion = np.concatenate([alpha_hist,phi_hist,theta_hist])
    
    return np.array(descripcion), np.array(extremValues)
    
###############################################################################
def getDescriptorSet(fpfhSet, pc, tamano_pc, NumPoints):

    #Cantidad de conjunto de puntos en una imagen
    cant_conjuntos = int(tamano_pc / NumPoints)
    
    pointSet = []
    indexSet = []
    extremSet = []
    
    #Por cada uno de los conjuntos
    for cant in range(cant_conjuntos):
    
        kdtree = pcl.KdTreeFLANN(pc)
        
        #punto random del conjunto de puntos
        punto = randint(0,pc.size - 1)
        
        #puntos cercanos al punto tomado
        nearPoint, d = kdtree.nearest_k_search_for_point(pc, punto, NumPoints)
        pc_arr = pc.to_array()
        
        #Descipcion y extremos del conjunto de puntos
        descripcion, extremos = getExtremAndDescription(pc_arr, fpfhSet,
                                                        nearPoint)
        
        #conjunto de decriptores
        pointSet.append(descripcion)
        
        #Conjunto de indices de los descriptores
        indexSet.append(nearPoint)
        
        #conjunto de puntos extremos de cada conjunto de descriptores
        extremSet.append(extremos)
        
        #Quita los puntos que se quitarán del fpfh
        notUsedInd = DescriptorUtils.erraseUsedPointInArray(pc.size,nearPoint)
        
        #Eliminar la seccion revisada para no volver a considerarlo
        nuevo_arr_pc = quitar_segmento(pc_arr,nearPoint)
        pc = pc.extract((nuevo_arr_pc))
        
        #Reorganiza los FPFH al nuevo pc
        fpfhSet = DescriptorUtils.createNewFPFH(fpfhSet, notUsedInd)
        
        #pc.to_file('parte_%d.pcd'% cont)
        
    return pointSet, indexSet, extremSet

###############################################################################
def inicio(fpfhSet, pc, NumPoints, verbose):
    
    if (verbose): print ("getDescriptorSet")
    fpfhSet, indexSet ,extremSet= getDescriptorSet(fpfhSet,
                                                   pc,
                                                   pc.size,
                                                   NumPoints)
    
    return fpfhSet, indexSet, extremSet

###############################################################################
'''
direccion = '../data/casi_final.pcd'

pc = pcl.PointCloud()
pc.from_file(direccion)
 
kdtree = pcl.KdTreeFLANN(pc)

cantidad_puntos = 1000

inicio(pc,cantidad_puntos)
'''


