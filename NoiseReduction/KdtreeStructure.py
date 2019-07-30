import pcl
import math
import numpy as np
  
#inicializa una pointCloud
def inicializar_pointCloud():
    return pcl.PointCloud()

###############################################################################
# OBTENER POINT CLOUD (PC)


#lee un archivo PCD y lo introduce a un PointCloud
    #pc = point cloud inicializado
    #direccion = direccion del archivo
def readPCDFile(pc,direccion):
    pc.from_file(direccion)

#se obtiene el pointCloud de la direccion otorgada
    #direccion = direccion del archivo
def getPointCloudFromDir(direccion):
    
    pc = inicializar_pointCloud()
    
    readPCDFile(pc,direccion)
    
    return pc

###############################################################################
# OBTENER KDTREE

#Se obtiene un kdtree a partir de un pointCloud
    #pc = Point cloud
def getKdtreeFromPointCloud(pc):
    return pcl.KdTreeFLANN(pc)

###############################################################################
# FUNCION INICIAL GENERAL
    
def getKdtreeFromPointCloudDir(direccion):
    
    rawPc = getPointCloudFromDir(direccion)
    
    pc = erraseEmptyCoords(rawPc)
    
    kdtree = getKdtreeFromPointCloud(pc)
    
    return pc, kdtree

###############################################################################
    

#Se obtiene un arreglo de kdtree a partir de un arreglo de pointCloud
    #num = cantidad de datos en el arreglo de Point cloud
    #arrpc = un arreglo de Point cloud
def getKdtreeFromPointCloudArr(num,arrpc):
    
    arrKdtree = []
    
    for cant in num:
        arrKdtree.append(getKdtreeFromPointCloud(arrpc[cant]))
    
    return arrKdtree


###############################################################################

#Se obtiene un kdtree de la direccion otorgada
def obtencion_kdtree(direccion):
    
    pc = getPointCloudFromDir(direccion)
    
    return pcl.KdTreeFLANN(pc)
   
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
#obtiene un conjunto de pointCloud del tamano 
    #Num = numero de frames a capturar
def obtencion_pointCloud_escena(num,direccion):
    
    arrPc = []
    
    for cant in range(num):
        arrPc.append(obtencion_pointCloud(direccion%(cant+1)))
    
    return arrPc
'''




###############################################################################
def erraseEmptyCoords(pc):
    
    pcArray = pc.to_array()
    
    notEmptyPoints = []
    
    for pos in range(pc.size):
        
        if not math.isnan(pcArray[pos][0]):
            
            notEmptyPoints.append(pcArray[pos])
    
    pcArray =  np.asarray(notEmptyPoints)
    
    return pcl.from_array(pcArray)