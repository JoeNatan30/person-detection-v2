#from lectura import lectura_de_kinect as lk
#from lectura import lectura_xml
from NoiseReduction import ReduceNoise, KdtreeStructure
from Segmentation import RansacAlgorithm
from Characterization import Fpfh, Descriptor, Procesados
from Algorithms import CrossValidation
from Algorithms import Svm, NeuralNetwork, RandomForest

import time
import pandas as pd

###############################################################################
#Definitions    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###############################################################################
'''
def lectura_(cant_PCD):
    lk.iniciar(cant_PCD)
'''
###############################################################################
def segmentation(pos, pc, kdtree, v, verbose):
    
    if (verbose): print ("RansacAlgorithm.iniciar")
    pc_sin_seg, kdtree_sin_seg = RansacAlgorithm.iniciar(pc,kdtree,v)
   
    #pc, kdtree = KdtreeStructure.getKdtreeFromPointCloudDir('data/entrenamiento/segmentado2/segmentado_%d.pcd'%(pos))
    
    if (verbose): print ("ReduceNoise.reduceDistancePoint")
    pc_sin_out = ReduceNoise.reduceDistancePoint(pc_sin_seg,kdtree_sin_seg,v)
    
    if (verbose): print ("KdtreeStructure.getKdtreeFromPointCloud")
    kdtree_sin_out = KdtreeStructure.getKdtreeFromPointCloud(pc_sin_out)
    
    if (verbose): print ("ReduceNoise.reduceDistancePoint")
    pc_sin_out2 = ReduceNoise.reduceDistancePoint(pc_sin_out,kdtree_sin_out,v)
    
    if (verbose): print ("KdtreeStructure.getKdtreeFromPointCloud")
    kdtree_sin_out2 = KdtreeStructure.getKdtreeFromPointCloud(pc_sin_out2)
    
    pc_sin_out2.to_file(str.encode('./segmentado_%d.pcd'%(pos)))
    #pc_sin_out2.to_file('data/entrenamiento/segmentado/segmentado_%d.pcd'%(pos))
    
    return pc_sin_out2,kdtree_sin_out2

###############################################################################
def histograma(pc, kdtree, verbose):
    #pc_seg, kdtree_seg = KdtreeStructure.getKdtreeFromPointCloudDir('data/entrenamiento/segmentado/segmentado_%d.pcd'%(pos))
    
    if (verbose): print ("Fpfh.inicio")
    list_fpfh_point = Fpfh.inicio(pc, kdtree, verbose)

    return list_fpfh_point

###############################################################################
def descripcion(fpfh, tamano, pc, verbose):

    if (verbose): print ("Descriptor.inicio")
    conj_fpfh, conj_ind, conj_extr = Descriptor.inicio(fpfh, pc, tamano)
    
    return conj_fpfh, conj_ind, conj_extr

###############################################################################
def descripcion_train(pos,tamano, fpfh,pc, kdtree, verbose):
    
    if (verbose): print ("Descriptor.obtener_descriptores_train")
    
    dataset_X, dataset_Y, conj_extre= Descriptor.obtener_descriptores_train(pos,
                                                                            fpfh,
                                                                            pc,
                                                                            kdtree,
                                                                            tamano)
    
    return dataset_X, dataset_Y, conj_extre

###############################################################################
def guardar_datos_procesados(pos,data_x, data_y):
    
    datos = {'es':data_y,'histograma':data_x}
            
    dtp = pd.DataFrame(datos)
            
    dtp.to_pickle('./procesado_%d.pkl'%pos)
    
    #para leer el fpfh de un archivo
    #fpfh = pd.read_pickle('data/entrenamiento/fpfh_%d.pkl'%pos)
    #fpfh_list = fpfh.values
    
###############################################################################
def procesamiento_train(cant_PCD, porcentaje, tamano, version,
                        max_paral, pos_paral, rangeOfDiff, verbose):
    
    rest = pos_paral
    
    tope = cant_PCD * porcentaje / 100
    
    for pos in range(cant_PCD):
        
        if(rest == pos % max_paral and pos < tope):
            
            print ("posición: ", pos)
            print("menor que: ",rangeOfDiff)
            #Ruido
            if (verbose): print ("ReduceNoise.ruido")
            pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                    rangeOfDiff,
                    pos,
                    verbose)
            
            print('Tamaño Sin Ruido: ',pc_sin_ruido.size)
            
            #segmentacion
            if (verbose): print ("segmentation")
            pc_seg, kdtree_seg = segmentation(pos,pc_sin_ruido,
                                              kdtree_sin_ruido,
                                              version, verbose)
            print (pc_seg.size)
            
            #FPFH
            if (verbose): print ("Histogram")
            #pc_seg, kdtree_seg = KdtreeStructure.getKdtreeFromPointCloudDir('data/entrenamiento/segmentado/segmentado_%d.pcd'%pos)
            fpfh_list = histograma(pc_seg, kdtree_seg, verbose)
            
            #Descriptor
            if (verbose): print ("Descriptor")
            dataset_X, dataset_Y, conj_extre = descripcion_train(pos,
                                                                 tamano, 
                                                                 fpfh_list,
                                                                 pc_seg,
                                                                 kdtree_seg,
                                                                 verbose)
            
            #Guardado
            if (verbose): print ("Save Processed Data")
            guardar_datos_procesados(pos,dataset_X,dataset_Y)
            
            print ("END - Process: ", pos)
            
###############################################################################
def procesamiento_real(cant_PCD, porcentaje, tamano, version,
                       max_paral, pos_paral, rangeOfDiff, verbose):
    
    rest = pos_paral
    
    base = cant_PCD * porcentaje / 100
    base -= 1
    
    for pos in range(cant_PCD):
         
        if(rest == pos % max_paral and pos > base):
        
            print (pos)
            
            #Ruido
            pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                    rangeOfDiff,
                    pos,
                    verbose)
            
            #segmentacion        
            pc_seg, kdtree_seg = segmentation(pos,pc_sin_ruido, kdtree_sin_ruido,version)
            
            #FPFH
            #pc_seg, kdtree_seg = KdtreeStructure.getKdtreeFromPointCloudDir('data/entrenamiento/segmentado/segmentado_%d.pcd'%pos)
            fpfh_list = histograma(pc_seg, kdtree_seg)
            
            #Descriptor
            dataset_X, dataset_Y, conj_extre = descripcion(fpfh_list,tamano,pc_seg)
            
            #Guardado
            guardar_datos_procesados(pos,dataset_X,dataset_Y)

###############################################################################
def entrenamiento(cant, porcentaje, algoritmo, verbose):

    data_x, data_y = Procesados.entrenamiento(cant,porcentaje)
    
    start_time = time.time()

    if algoritmo == "svm":
        
        print ("inicio entrenamiento SVM")
        Svm.entrenar(data_x,data_y)
        print ("SVM entrenado",("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio entrenamiento redes neuronales")
        NeuralNetwork.entrenar(data_x,data_y)
        print ("Redes Neuronales entrenado", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio Random Forest")
        RandomForest.entrenar(data_x,data_y)
        print ("Random Forest entrenado", ("--- %s seconds ---" % (time.time() - start_time)))
        
###############################################################################
def validacion_cruzada(cant, porcentaje, algoritmo, verbose):
    
    data_x , data_y = Procesados.prueba(cant,porcentaje)
    
    start_time = time.time()
    
    if algoritmo == "rn":
        
        print ("inicio cross validation SVM")
        CrossValidation.optimize_svc(data_x,data_y)
        print ("fin cross validation SVM", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio cross validation Redes Neuronales")
        CrossValidation.optimize_nn(data_x,data_y)

        print ("fin cross validation Redes Neuronales",("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio cross validation Forest")
        CrossValidation.optimize_rfc(data_x,data_y)
        print ("fin cross validation Random Forest", ("--- %s seconds ---" % (time.time() - start_time)))
        
###############################################################################    
def prueba(cant, porcentaje, algoritmo, verbose):
    
    data_x , data_y = Procesados.prueba(cant,porcentaje)
    
    start_time = time.time()
    
    if algoritmo == "svm":
        
        print ("inicio prediccion SVM")
        Svm.predecir(data_x,data_y)
        print ("fin prediccion SVM", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio prediccion Redes Neuronales")
        NeuralNetwork.predecir(data_x,data_y)
        print ("fin prediccion Redes Neuronales", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio Random Forest")
        RandomForest.predecir(data_x,data_y)
        print ("fin prediccion Random Forest", ("--- %s seconds ---" % (time.time() - start_time)))
  
###############################################################################
def medition(cant_PCD,tamano_conjunto, version,
             max_paral, pos_paral,rangeOfDiff,tipo,verbose):
    
    print ("medición de " + tipo)
    for pos in range(cant_PCD):
        
        ReduceNoise.medition(rangeOfDiff,pos,tipo,verbose)