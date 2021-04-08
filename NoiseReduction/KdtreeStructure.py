import math
import numpy as np
import open3d as o3d


###############################################################################
# FUNCION INICIAL GENERAL
def getKdtreeFromPointCloudDir(direccion):

    rawPc = getPointCloudFromDir(direccion)

    pc, size = erraseEmptyCoords(rawPc)

    kdtree = getKdtreeFromPointCloud(pc)

    return pc, kdtree, size


###############################################################################
# inicializa una pointCloud
def inicializar_pointCloud():
    return o3d.geometry.PointCloud()

###############################################################################
# OBTENER POINT CLOUD (PC)


# lee un archivo PCD y lo introduce a un PointCloud
    # pc = point cloud inicializado
    # direccion = direccion del archivo
def readPCDFile(direccion):

    return o3d.io.read_point_cloud(direccion)


# se obtiene el pointCloud de la direccion otorgada
    # direccion = direccion del archivo
def getPointCloudFromDir(direccion):

    pcd = readPCDFile(direccion)

    return pcd


###############################################################################
# OBTENER KDTREE
# Se obtiene un kdtree a partir de un pointCloud
# pc = Point cloud
def getKdtreeFromPointCloud(pcd):

    return o3d.geometry.KDTreeFlann(pcd)


###############################################################################

# Se obtiene un arreglo de kdtree a partir de un arreglo de pointCloud
# num = cantidad de datos en el arreglo de Point cloud
# arrpc = un arreglo de Point cloud
def getKdtreeFromPointCloudArr(num, arrpc):

    arrKdtree = []

    for cant in num:
        arrKdtree.append(getKdtreeFromPointCloud(arrpc[cant]))

    return arrKdtree


###############################################################################
# Se obtiene un kdtree de la direccion otorgada
def obtencion_kdtree(direccion):

    pcd = getPointCloudFromDir(direccion)

    return o3d.geometry.KDTreeFlann(pcd)


'''
#obtiene un conjunto de kdtree del tamano otorgado
    #Num = numero de frames a capturar
def obtencion_kdtree_escena(num,direccion):

    arrKdtree = []

    for cant in range(num):
        arrKdtree.append(obtencion_kdtree(direccion%(cant+1)))

    return arrKdtree
'''

'''
# obtiene un conjunto de pointCloud del tamano
    #Num = numero de frames a capturar
def obtencion_pointCloud_escena(num,direccion):

    arrPc = []

    for cant in range(num):
        arrPc.append(obtencion_pointCloud(direccion%(cant+1)))

    return arrPc
'''


###############################################################################
def erraseEmptyCoords(pcd):

    pcArray = np.asarray(pcd.points)

    notEmptyPoints = []

    for point in pcArray:

        if not math.isnan(point[0]):

            notEmptyPoints.append(point)

    xyz = np.asarray(notEmptyPoints)
    size = len(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, size
