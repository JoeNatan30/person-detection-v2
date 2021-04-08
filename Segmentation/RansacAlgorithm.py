import numpy as np
import math
from cmd import Cmd
from Segmentation import RansacUtils


###############################################################################
def removeSegment(arr_general, plane_pos):

    nuevo_arr = []

    """
    For each point of the general array

    if a point of the plane is in this array, these point is ignored
    """
    for pos in plane_pos:
        nuevo_arr.append(arr_general[pos])

    return nuevo_arr


###############################################################################
def isWithinVolume(point, f_x, rango):

    # d = |A_x * p_0 + B_y * p_1 + C_z * p_2 + D|
    #     ---------------------------------------
    #               Raiz(A^2+B^2+C^2)

    # A_X * p_0
    A_x0 = f_x[0]*point[0]

    # B_Y * p_1
    B_y0 = f_x[1]*point[1]

    # C_Z * p_2
    C_z0 = f_x[2]*point[2]

    D = f_x[3]

    # |A_x0 + B_y0 + C_z0 + D|
    dividendo = RansacUtils.absolute(A_x0 + B_y0 + C_z0 + D)

    # because A^2+B^2+C^2 = (A+B+C)^2 - 2 * (AB+BC+AC)
    # divisor = f_x[0]**2 + f_x[1]**2 + f_x[2]**2
    divisor = (f_x[0]+f_x[1]+f_x[2])**2 - 2*(f_x[0]*f_x[1] +
                                             f_x[1]*f_x[2] +
                                             f_x[0]*f_x[2])

    if divisor == 0:
        divisor = 0.000000000001

    distancia = dividendo / (divisor**(1/2))

    if float(distancia) < float(rango):
        return True
    else:
        return False


###############################################################################
def isRepeatedPoint(ramdomPoints, punto):

    for point in ramdomPoints:
        if point == punto:
            return True

    return False


###############################################################################
def getNormal(punto1, punto2, punto3):

    p3p1 = punto1 - punto3
    p3p2 = punto2 - punto3

    # producto crus
    return np.cross(p3p1, p3p2)


###############################################################################
def getFalsePoint(pc,pc_arr,listOfPoints,kdtree):
    
    numNearPoints = 11
    
    pos_puntos , distancia = kdtree.nearest_k_search_for_point(pc
                                                               ,listOfPoints[0]
                                                               ,numNearPoints)
   
    punto_falso = np.zeros(3)
    
    for pos in pos_puntos:
        punto_falso = punto_falso + pc_arr[pos]
   
    punto_falso = punto_falso/numNearPoints
    
    return punto_falso


###############################################################################
def getFunctionOfPlane(pc, pc_arr, listOfPoints, kdtree):

    punto_1 = pc_arr[listOfPoints[0]]
    punto_2 = pc_arr[listOfPoints[1]]
    punto_3 = pc_arr[listOfPoints[2]]

    # Calcular la normal a partir de 3 puntos
    normal = getNormal(punto_1, punto_2, punto_3)

    plane_function = RansacUtils.assignArray3d(normal)

    # Pl_P . n = 0
    p1p_n = RansacUtils.negative(punto_1) * normal

    d = 0

    # La suma de la parte d de "Ax + By + Cz + d"
    for comp in p1p_n:
        d = d + comp

    plane_function.append(d)

    # contiene los valores de A, B, C y D
    return plane_function


###############################################################################
def randomPoints(total):

    rand_1 = RansacUtils.randomIntBetween(0, total-1)
    rand_2 = RansacUtils.randomIntBetween(0, total-1)
    rand_3 = RansacUtils.randomIntBetween(0, total-1)

    # To avoid reapeted numbers
    while (rand_1 == rand_2 or rand_1 == rand_3 or rand_2 == rand_3):
        rand_2 = RansacUtils.randomIntBetween(0, total-1)
        rand_3 = RansacUtils.randomIntBetween(0, total-1)

    resultado = RansacUtils.assignTo3dArray(rand_1, rand_2, rand_3)

    return resultado


###############################################################################
def ransac(pcd, kdtree, rango):

    pc_arr = RansacUtils.getArrFromPcd(pcd.points)

    # Variables fijas
    numInteractions = 0
    bestMeasure = 0
    bestVolumeArr = []
    totalPoints = len(pc_arr)

    # rango de distancia cuadratica
    # rango **= 2

    # This value is used only to initialize K it could be any value over 1
    k = 1000

    # la cantidad de veces que se hara el ciclo
    while numInteractions < k:

        puntos_azar = randomPoints(totalPoints)

        # Se obtiene ademas el valor "d" de la funcion del plano
        f_x = getFunctionOfPlane(pcd, pc_arr, puntos_azar, kdtree)

        withinVolumenArr = []

        # Para saber si un punto esta dentro del volumen limite del plano
        for pos, point in enumerate(pc_arr):

            # Si el punto no se repite con los dos elegidos al azar
            if not isRepeatedPoint(puntos_azar, pos):

                # Si esta dentro del rango que forma un cubo
                if isWithinVolume(point, f_x, rango):
                    withinVolumenArr.append(pos)

        # Para obtener la cantidad de puntos dentro del cubo
        pointsInVolume = len(withinVolumenArr)

        # Obtener la mayor cantidad de puntos de todas las pruebas
        if (pointsInVolume > bestMeasure):

            # Statistic Formula to get the best approximate value of iteration
            k = int(math.log(1 - 0.9999) /
                    math.log(1 - (pointsInVolume/totalPoints)**3))

            print("El mejor valor es: ", pointsInVolume,
                  " Max iter k = ", k,
                  " ahora en: ", numInteractions)

            bestMeasure = pointsInVolume
            bestVolumeArr = withinVolumenArr

        numInteractions += 1

    # Se le quita el segmento del total de puntos
    # (bestVolumenArr = array of points)

    print("before extract: ", totalPoints)

    pcd_segmentado = RansacUtils.errasePointInPcd(pc_arr, bestVolumeArr)

    print("after extract: ", len(np.asarray(pcd_segmentado.points)))

    kdtree_segmentado = RansacUtils.getKdtreeFromPointCloud(pcd_segmentado)
    #RansacUtils.showPoints(pcd_segmentado)
    # pc_segmento = pc.extract(mejor_volumen)
    # pc_segmento.to_file('../data/segmento/seg.pcd')

    return pcd_segmentado, kdtree_segmentado


###############################################################################
# Main
###############################################################################
def iniciar(pcd, kdtree, v):

    if v == 1:
        # Variable de precision
        rango = 0.7
        
    else:
        rango = 0.08

    print("rango ransac = ", rango)

    newPc, newKdtree = ransac(pcd, kdtree, rango)

    return newPc, newKdtree


###############################################################################
# Medition
###############################################################################
def getPlaneSectionPcd(arr, rango):

    # Variables fijas
    numInteractions = 0
    bestMeasure = 0
    bestVolumeArr = []
    totalPoints = len(arr)

    # rango de distancia cuadratica
    # rango **= 2

    # This value is used only to initialize K it could be any value over 1
    k = 1000

    # la cantidad de veces que se hara el ciclo
    while numInteractions < k:

        puntos_azar = randomPoints(totalPoints)
        # Se obtiene ademas el valor "d" de la funcion del plano
        f_x = getFunctionOfPlane([], arr, puntos_azar, [])

        withinVolumenArr = []

        # Para saber si un punto esta dentro del volumen limite del plano
        for pos, point in enumerate(arr):

            # Si el punto no se repite con los dos elegidos al azar
            if not isRepeatedPoint(puntos_azar, pos):

                # Si esta dentro del rango que forma un cubo
                if isWithinVolume(point, f_x, rango):
                    withinVolumenArr.append(point)

        # Para obtener la cantidad de puntos dentro del cubo
        pointsInVolume = len(withinVolumenArr)

        # Obtener la mayor cantidad de puntos de todas las pruebas
        if (pointsInVolume > bestMeasure):

            # Statistic Formula to get the best approximate value of iteration
            k = int(math.log(1 - 0.999999) /
                    math.log(1 - (pointsInVolume/totalPoints)**3))

            print("El mejor valor es: ", pointsInVolume,
                  " Max iter k = ", k,
                  " ahora en: ", numInteractions)

            bestMeasure = pointsInVolume
            bestVolumeArr = withinVolumenArr

        numInteractions += 1

    # Se le quita el segmento del total de puntos
    # (bestVolumenArr = array of points)

    # pcd_segmentado = RansacUtils.getPointInPcd(arr, bestVolumeArr)
    # pcd_segmentado_err = RansacUtils.errasePointInPcd(arr, bestVolumeArr)

    print("after extract: ", len(np.asarray(bestVolumeArr)))

    # kdtree_segmentado = RansacUtils.getKdtreeFromPointCloud(pcd_segmentado)

    # pc_segmento = pc.extract(mejor_volumen)
    # pc_segmento.to_file('../data/segmento/seg.pcd')

    return bestVolumeArr


def medition(pcd, kdtree, verbose):

    rangeList = np.linspace(0.2, 1.7, 31)
    pc_arr = RansacUtils.getArrFromPcd(pcd.points)

    plane_arr = []
    for point in pc_arr:
        if(point[1] < -15):
            plane_arr.append(point)
    # plane_arr = getPlaneSectionPcd(pc_arr, 0.8)

    total = len(plane_arr)
    resultList = []

    for value in rangeList:
        part_arr = getPlaneSectionPcd(plane_arr, value)
        resultList.append(len(part_arr)/total)

    return resultList


###############################################################################
# Code - main
###############################################################################


'''
pc = pcl.PointCloud()
#pc.from_file('../data/sin_ruido/sin_plano/005_5.pcd')
#pc.from_file('../data/sin_ruido/sin_plano/sin_plano.pcd')
pc.from_file('../data/segmento/sin_segmento5.pcd')
kdtree = pcl.KdTreeFLANN(pc)
arr_plano, arr_general = iniciar(pc,kdtree)
nuevo_arr = quitar_segmento(arr_general,arr_plano)
pc_sin_segmento = pc.extract(nuevo_arr)
pc_sin_segmento.to_file('../data/segmento/sin_segmento6.pcd')
'''

'''
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits
        well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good
                                                        model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points
                      in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
'''
