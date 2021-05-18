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
def spfh(kdtree, normal_arr, points_arr, cant):

    spfhPoint = []
    theta_comp_1 = 0.0
    theta_comp_2 = 0.0

    # Position in the taken point cloud
    for pos, point in enumerate(points_arr):

        if (pos % 10000 == 0):
            print(pos, "iteraciones en spfh")

        # Calcular puntos adyacentes
        _, nearPoint, distance = kdtree.search_knn_vector_3d(point, cant)

        # Inicializar el arreglo de componentes para cada punto con valor 0
        spfhComponents = np.zeros(3)

        # Point's number considered in the SPFH operation (main point included)
        numberOfPointsConsidered = 1.0

        # ind = posicion en el arreglo de puntos_cercano
        for ind, closePointPos in enumerate(nearPoint):

            # To avoid taking the same point two times
            if(ind == 0):
                continue

            # To avoid taking a far point
            if(distance[ind] < 3.0):

                numberOfPointsConsidered += 1

                ########################
                # Select source and tarjet

                # First - get the angle of the point "pos" with its normal
                pppi = points_arr[closePointPos] - point
                pppi_normalized = FpfhUtils.vectorNormalization(pppi)

                n_p_normalized = FpfhUtils.vectorNormalization(
                    normal_arr[pos])

                # p = pos
                angle_p = FpfhUtils.angleBetweenTwoVectors(n_p_normalized,
                                                           pppi_normalized)

                # Second - get the angle of the point "ind" with its normal
                pipp = point - points_arr[closePointPos]
                pipp_normalized = FpfhUtils.vectorNormalization(pipp)

                n_i_normalized = FpfhUtils.vectorNormalization(
                    normal_arr[closePointPos])

                # i = ind
                angle_i = FpfhUtils.angleBetweenTwoVectors(n_i_normalized,
                                                           pipp_normalized)

                # Compare both angle to get to know
                # the point source and target
                if(angle_p <= angle_i):

                    p_s = point
                    p_t = points_arr[closePointPos]

                    n_s = normal_arr[pos]
                    n_t = normal_arr[closePointPos]

                else:

                    p_s = points_arr[closePointPos]
                    p_t = point

                    n_s = normal_arr[closePointPos]
                    n_t = normal_arr[pos]

                ########################
                # FPFH Operations SECTION

                # Both normals normalized
                n_s_normalized = FpfhUtils.vectorNormalization(n_s)
                n_t_normalized = FpfhUtils.vectorNormalization(n_t)

                # U vector (normalized)
                u_normalized = n_s_normalized  # u = n_s

                # Distance vector between the actual point & the near point
                pspt = p_t - p_s  # P_t - P_s
                pspt_normalized = FpfhUtils.vectorNormalizationNorm2(pspt)

                # V vector
                # v = (P_t - P_s) x u
                v = np.cross(pspt_normalized, u_normalized)
                v_normalized = FpfhUtils.vectorNormalization(v)

                # W vector
                # w = u x v
                w = np.cross(u_normalized, v_normalized)
                w_normalized = FpfhUtils.vectorNormalization(w)

                # Alpha angle
                alpha = FpfhUtils.angleBetweenTwoVectors(v_normalized,
                                                         n_t_normalized)

                # phi angle
                phi = FpfhUtils.angleBetweenTwoVectors(u_normalized,
                                                       pspt_normalized)

                # Theta angle
                # comp_1 = w . n_t
                theta_comp_1 = np.dot(w_normalized, n_t_normalized)
                # comp_2 = u . n_t
                theta_comp_2 = np.dot(u_normalized, n_t_normalized)
                # theta = arctan( w . n_t ; u . n_t)
                theta = math.atan2(theta_comp_1, theta_comp_2)

                alpha = abs(alpha)
                phi = abs(phi)
                theta = abs(theta)

                # To save SPFH's components (accumulate)
                spfhComponents[0] += alpha
                spfhComponents[1] += phi
                spfhComponents[2] += theta

        # Division entre la cantidad de puntos adyacentes
        spfhComponents /= numberOfPointsConsidered

        # To save point's components
        spfhPoint.append(spfhComponents)

    # Se retorna la lista de los puntos
    return spfhPoint


###############################################################################
def fpfh(kdtree, list_spfh, pointArr, cant):

    list_fpfh = []

    # Position in the taken point cloud
    for pos, point in enumerate(pointArr):

        if (pos % 10000 == 0):
            print(pos, "iteraciones en fpfh")
        _, nearPointList, distance = kdtree.search_knn_vector_3d(point, cant)

        fpfh = np.zeros(3)
        spfh_general_adyacente = np.zeros(3)

        # Number of point considered in FPFH formula (main point not included)
        numberOfPointsConsidered = 0.0

        # Calculo de FPFH del punto objetivo
        for ind, nearPointPos in enumerate(nearPointList):

            # To avoid taking the same point two times
            if ind == 0:
                continue

            # To avoid taking a far point
            if(distance[ind] < 3.0):

                numberOfPointsConsidered += 1

                '''
                Exponential is used to have better ponderation because
                distance value is frequently less than 1
                '''
                print("distancia: %f", math.sqrt(math.exp(distance[ind])))
                weigh = math.sqrt(math.exp(distance[ind]))

                nearSpfh = np.array(list_spfh[nearPointPos])

                # ponder components of spfh
                spfh_general_adyacente += nearSpfh/weigh

        # To avoid a zero division
        if(numberOfPointsConsidered != 0.0):
            # Division entre la cantidad de puntos adyacentes tomados
            spfh_general_adyacente /= numberOfPointsConsidered

        # Sum spfh of the point with the average of the
        # spfh of the taken near points
        fpfh += list_spfh[pos]
        fpfh += spfh_general_adyacente

        # modify fpfh components to have it normalized
        # fpfh = FpfhUtils.componentsNormalization(fpfh)

        # add to list FPFH
        list_fpfh.append(fpfh)

    # Se retorna la lista fpfh y su respectivo indice en la nube de puntos
    return np.array(list_fpfh)


###############################################################################
# MAIN
###############################################################################
def inicio(pcd, kdtree, tamano, verbose):

    # puntos (salen 2 normales menos al valor indicado en cantidad)
    quantity = 120
    cantidad_fpfh = tamano
    radious = 5

    if (verbose):
        print("normal.getNormalDirection")

    pcdNb = FpfhUtils.getEstimatedNormals(pcd, quantity)

    pcdN = FpfhUtils.fixNormalDirectionInPCD(pcdNb)

    #FpfhUtils.showNormals(pcdN)
    #pcdN = normal.getNormalDirection(pcd, kdtree, cantidad)

    normalArr = np.asarray(pcdN.normals)
    pointArr = np.asarray(pcdN.points)

    point_fpfh = FpfhUtils.fpfh(pcd, cantidad_fpfh, radious)

    localfeature = point_fpfh.data

    #for val in fpfhArr.T:
    #    print(val)

    '''
    if (verbose):
        print("SPFH")

    listSpfhPoint = spfh(kdtree, normalArr, pointArr, cantidad_fpfh)

    if (verbose):
        print("FPFH")

    arr_point_fpfh = fpfh(kdtree, listSpfhPoint, pointArr, cantidad_fpfh)
    '''
    return localfeature.T
###############################################################################


###############################################################################
# CODE
###############################################################################
'''
pc = pcl.PointCloud()
pc.from_file('../data/segmento2/sin_segmento6.pcd')
kdtree = pcl.KdTreeFLANN(pc)

inicio(pc,kdtree)
'''
