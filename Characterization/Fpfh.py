#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:59:35 2018

@author: joe
"""
from Characterization import Normal_fpfh as normal
from Characterization import FpfhUtils
import numpy as np
import math

###############################################################################
# Modules
###############################################################################
def spfh(pc, kdtree, normal_arr, num_puntos, cant):
    
    pc_arr = pc.to_array()
    
    spfhPoint = []

    theta_comp_1 = 0.0
    theta_comp_2 = 0.0
    
    #Position in the taken point cloud 
    for pos in range(num_puntos):

        if (pos % 10000 == 0): print (pos, "iteraciones en spfh")
        #Calcular puntos adyacentes
        punto_cercano, distancia = kdtree.nearest_k_search_for_point(pc,pos,
                                                                     cant)
        
        #Inicializar el arreglo de componentes para cada punto con valor 0        
        spfhComponents = np.zeros(3)
        
        #Number of point considered in the SPFH operation (main point included)
        numberOfPointsConsidered = 1.0

        # ind = posicion en el arreglo de puntos_cercano
        for ind in range(len(punto_cercano)):
            
            #To avoid taking the same point two times
            if(ind > 0): 
                
                #To avoid taking a far point
                if(distancia[ind] < 3.0): 
                    
                    numberOfPointsConsidered += 1
                    pos_adyacente = punto_cercano[ind]
                    
                    ########################
                    #Select source and tarjet
                    
                    #First - get the angle of the point "pos" with its normal
                    pppi = pc_arr[pos_adyacente] - pc_arr[pos]
                    pppi_normalized = FpfhUtils.vectorNormalization(pppi)
                    
                    n_p_normalized = FpfhUtils.vectorNormalization(normal_arr[pos])
                    
                    angle_p = FpfhUtils.angleBetweenTwoVectors(n_p_normalized, pppi_normalized) # p = pos
                    
                    #Second - get the angle of the point "ind" with its normal
                    pipp = pc_arr[pos] - pc_arr[pos_adyacente]
                    pipp_normalized = FpfhUtils.vectorNormalization(pipp)
                    
                    n_i_normalized = FpfhUtils.vectorNormalization(normal_arr[pos_adyacente])
                    
                    angle_i = FpfhUtils.angleBetweenTwoVectors(n_i_normalized, pipp_normalized) # i = ind
                    
                    #Compare both angle to get to know the point source and target
                    if(angle_p <= angle_i):
                        
                        p_s = pc_arr[pos]
                        p_t = pc_arr[pos_adyacente]
                        
                        n_s = normal_arr[pos]
                        n_t = normal_arr[pos_adyacente]
                    
                    else:
                        
                        p_s = pc_arr[pos_adyacente]
                        p_t = pc_arr[pos]
                        
                        n_s = normal_arr[pos_adyacente]
                        n_t = normal_arr[pos]
                        
                    ########################
                    #FPFH Operations SECTION
                    
                    #Both normals normalized
                    n_s_normalized = FpfhUtils.vectorNormalization(n_s)
                    n_t_normalized = FpfhUtils.vectorNormalization(n_t)
                    
                    #U vector (normalized)
                    u_normalized = n_s_normalized  # u = n_s
                    
                    #Distance vector between the actual point an the near point
                    pspt = p_t - p_s  # P_t - P_s
                    pspt_normalized = FpfhUtils.vectorNormalizationNorm2(pspt)

                    #V vector                    
                    #v = (P_t - P_s) x u
                    v = np.cross(pspt_normalized,u_normalized) 
                    v_normalized = FpfhUtils.vectorNormalization(v)
                    
                    #W vector
                    w = np.cross(u_normalized,v_normalized) # w = u x v
                    w_normalized = FpfhUtils.vectorNormalization(w)
                    
                    
                    
                    #Alpha angle
                    alpha = FpfhUtils.angleBetweenTwoVectors(
                            v_normalized,
                            n_t_normalized)
                      
                    #phi angle
                    phi = FpfhUtils.angleBetweenTwoVectors(
                            u_normalized,pspt_normalized)

                    #Theta angle
                    theta_comp_1 = np.dot(w_normalized,n_t_normalized) #comp_1 = w . n_t
                    theta_comp_2 = np.dot(u_normalized,n_t_normalized) #comp_2 = u . n_t
                    
                    theta = math.atan2(theta_comp_1,theta_comp_2) # theta = arctan( w . n_t ; u . n_t)
                    
                    alpha = abs(alpha)
                    phi   = abs(phi)
                    theta = abs(theta)
                    
                    #To save SPFH's components (accumulate)
                    spfhComponents[0] += alpha
                    spfhComponents[1] += phi
                    spfhComponents[2] += theta
            
        #Division entre la cantidad de puntos adyacentes      
        spfhComponents /= numberOfPointsConsidered
        
        #To save point's components 
        spfhPoint.append(spfhComponents)

    #Se retorna la lista de los puntos y su indice
    return spfhPoint, num_puntos 

###############################################################################
def fpfh(pc, kdtree, list_spfh, num_puntos, cant):
    
   
    list_fpfh = []
    
    #Position in the taken point cloud
    for pos in range(num_puntos):
        
        if (pos % 10000 == 0): print (pos,"iteraciones en fpfh")
        nearPoint, distancia = kdtree.nearest_k_search_for_point(pc,pos,
                                                                     cant)
        
        fpfh = np.zeros(3)
        spfh_general_adyacente = np.zeros(3)
        
        #Number of point considered in FPFH formula (main point not included)
        numberOfPointsConsidered = 0.0
        
        #Calculo de FPFH del punto objetivo
        for ind in range(len(nearPoint)):
            
            #To avoid taking the same point two times
            if(ind>0):
                
                 #To avoid taking a far point
                if(distancia[ind] < 3.0): 
                
                    numberOfPointsConsidered += 1
                        
                    pos_adyacente = nearPoint[ind]
                        
                    '''
                    Exponential is used to have better ponderation because
                    distance value is frequently less than 1
                    '''
                    weigh = math.sqrt(math.exp(distancia[ind]))
                    
                    nearSpfh = np.array(list_spfh[pos_adyacente])
                        
                    #ponder components of spfh
                    spfh_general_adyacente += nearSpfh/weigh        
        
        #To avoid a zero division
        if(numberOfPointsConsidered != 0.0):
            #Division entre la cantidad de puntos adyacentes tomados
            spfh_general_adyacente /= numberOfPointsConsidered
            
        #Sum spfh of the point with the average of the spfh of the taken near points
        fpfh += list_spfh[pos]
        fpfh += spfh_general_adyacente
        
        #modify fpfh components to have it normalized
        #fpfh = FpfhUtils.componentsNormalization(fpfh)
     
        #add to list FPFH
        list_fpfh.append(fpfh)
        
    #Se retorna la lista fpfh y su respectivo indice en la nube de puntos
    return np.array(list_fpfh)

    
###############################################################################
#MAIN
###############################################################################
def inicio(pc, kdtree, verbose):

    cantidad = 80 #puntos (salen 2 normales menos al valor indicado en cantidad)
    cantidad_fpfh = 40
   
    if (verbose): print ("normal.getNormalDirection")
    normalArr, normalIndex = normal.getNormalDirection(pc,kdtree,cantidad)
    
    if (verbose): print ("SPFH")
    listSpfhPoint, numPoints = spfh(pc, kdtree, normalArr, len(normalIndex),
                                          cantidad_fpfh)
    if (verbose): print ("FPFH")
    arr_point_fpfh = fpfh(pc, kdtree, listSpfhPoint, numPoints, cantidad_fpfh)
    
    #
    
    return arr_point_fpfh
###############################################################################


###############################################################################
#CODE
###############################################################################
'''
pc = pcl.PointCloud()
pc.from_file('../data/segmento2/sin_segmento6.pcd')
kdtree = pcl.KdTreeFLANN(pc)

inicio(pc,kdtree)
'''