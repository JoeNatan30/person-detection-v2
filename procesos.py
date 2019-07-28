#from lectura import lectura_de_kinect as lk
#from lectura import lectura_xml
from eliminacion_de_ruido import estructura
from eliminacion_de_ruido import eliminar_ruido
from segmentacion import algoritmo_ransac
from caracterizacion import fpfh, descriptor, procesados
from algoritmos import cross_validation
from algoritmos import svm, neural_network, random_forest

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
def segmento(pos,pc,kdtree,v):
    
    pc_sin_seg, kdtree_sin_seg = algoritmo_ransac.iniciar(pc,kdtree,v)
    #pc_sin_seg.to_file('data/entrenamiento/segmentado2/segmentado_%d.pcd'%(pos))
    #pc, kdtree = estructura.obtencion_pointCloud_Kdtree('data/entrenamiento/segmentado2/segmentado_%d.pcd'%(pos))

    pc_sin_out = eliminar_ruido.sin_puntos_lejanos_distancias(pc_sin_seg,
                                                              kdtree_sin_seg,v)

    kdtree_sin_out = estructura.obtencion_kdtree_from_pointCloud(pc_sin_out)
    #
    pc_sin_out2 = eliminar_ruido.sin_puntos_lejanos_distancias(pc_sin_out,
                                                               kdtree_sin_out,
                                                               v)
    
    kdtree_sin_out2 = estructura.obtencion_kdtree_from_pointCloud(pc_sin_out2)
    #
    #pc_sin_out2.to_file('data/entrenamiento/segmentado/segmentado_%d.pcd'%(pos))
    
    return pc_sin_out2,kdtree_sin_out2

###############################################################################
def histograma(pc,kdtree):
    #pc_seg, kdtree_seg = estructura.obtencion_pointCloud_Kdtree('data/entrenamiento/segmentado/segmentado_%d.pcd'%(pos))
        
    list_fpfh_point = fpfh.inicio(pc,kdtree)

    return list_fpfh_point

###############################################################################
def descripcion(fpfh,tamano, pc):

    conj_fpfh, conj_ind, conj_extr = descriptor.inicio(fpfh, pc, tamano)
    
    return conj_fpfh, conj_ind, conj_extr

###############################################################################
def descripcion_train(pos,tamano, fpfh,pc, kdtree):
    
    dataset_X, dataset_Y, conj_extre= descriptor.obtener_descriptores_train(pos,fpfh,pc, kdtree,
                                                         tamano)
    return dataset_X, dataset_Y, conj_extre

###############################################################################
def guardar_datos_procesados(pos,data_x, data_y):
    
    datos = {'es':data_y,'histograma':data_x}
            
    dtp = pd.DataFrame(datos)
            
    dtp.to_pickle('data/entrenamiento/procesado/procesado_%d.pkl'%pos)
    
    #para leer el fpfh de un archivo
    #fpfh = pd.read_pickle('data/entrenamiento/fpfh_%d.pkl'%pos)
    #fpfh_list = fpfh.values
    
###############################################################################
def procesamiento_train(cant_PCD, porcentaje,tamano,version,max_paral,pos_paral):
    
    rest = pos_paral
    
    tope = cant_PCD * porcentaje / 100
    
    for pos in range(cant_PCD):
        
        if(rest == pos % max_paral and pos < tope):
        
            print (pos)
            
            #Ruido
            pc_sin_ruido, kdtree_sin_ruido = eliminar_ruido.ruido(pos)
            print ("fin sin ruido")
            
            #segmentacion        
            pc_seg, kdtree_seg = segmento(pos,pc_sin_ruido, kdtree_sin_ruido,version)
            print ("fin Segmentacion")
            
            #FPFH
            #pc_seg, kdtree_seg = estructura.obtencion_pointCloud_Kdtree('data/entrenamiento/segmentado/segmentado_%d.pcd'%pos)
            fpfh_list = histograma(pc_seg, kdtree_seg)
            
            #Descriptor
            dataset_X, dataset_Y, conj_extre = descripcion_train(pos,tamano,fpfh_list,pc_seg,kdtree_seg)
            
            #Guardado
            guardar_datos_procesados(pos,dataset_X,dataset_Y)
   
###############################################################################
def procesamiento_real(cant_PCD, porcentaje,tamano,version,max_paral,pos_paral):
    
    rest = pos_paral
    
    base = cant_PCD * porcentaje / 100
    base -= 1
    
    for pos in range(cant_PCD):
         
        if(rest == pos % max_paral and pos > base):
        
            print (pos)
            
            #Ruido
            pc_sin_ruido, kdtree_sin_ruido = eliminar_ruido.ruido(pos)
            
            #segmentacion        
            pc_seg, kdtree_seg = segmento(pos,pc_sin_ruido, kdtree_sin_ruido,version)
            
            #FPFH
            #pc_seg, kdtree_seg = estructura.obtencion_pointCloud_Kdtree('data/entrenamiento/segmentado/segmentado_%d.pcd'%pos)
            fpfh_list = histograma(pc_seg, kdtree_seg)
            
            #Descriptor
            dataset_X, dataset_Y, conj_extre = descripcion(fpfh_list,tamano,pc_seg)
            
            #Guardado
            guardar_datos_procesados(pos,dataset_X,dataset_Y)

###############################################################################
def entrenamiento(cant,porcentaje,algoritmo):

    data_x, data_y = procesados.entrenamiento(cant,porcentaje)
    
    start_time = time.time()

    if algoritmo == "svm":
        
        print ("inicio entrenamiento SVM")
        svm.entrenar(data_x,data_y)
        print ("SVM entrenado",("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio entrenamiento redes neuronales")
        neural_network.entrenar(data_x,data_y)
        print ("Redes Neuronales entrenado", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio Random Forest")
        random_forest.entrenar(data_x,data_y)
        print ("Random Forest entrenado", ("--- %s seconds ---" % (time.time() - start_time)))
        
###############################################################################
def validacion_cruzada(cant,porcentaje,algoritmo):
    
    data_x , data_y = procesados.prueba(cant,porcentaje)
    
    start_time = time.time()
    
    if algoritmo == "svm":
        
        print ("inicio cross validation SVM")
        cross_validation.optimize_svc(data_x,data_y)
        print ("fin cross validation SVM", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio cross validation Redes Neuronales")
        cross_validation.optimize_nn(data_x,data_y)

        print ("fin cross validation Redes Neuronales",("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio cross validation Forest")
        cross_validation.optimize_rfc(data_x,data_y)
        print ("fin cross validation Random Forest", ("--- %s seconds ---" % (time.time() - start_time)))
        
###############################################################################    
def prueba(cant,porcentaje,algoritmo):
    
    data_x , data_y = procesados.prueba(cant,porcentaje)
    
    start_time = time.time()
    
    if algoritmo == "svm":
        
        print ("inicio prediccion SVM")
        svm.predecir(data_x,data_y)
        print ("fin prediccion SVM", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rn":
        
        print ("inicio prediccion Redes Neuronales")
        neural_network.predecir(data_x,data_y)
        print ("fin prediccion Redes Neuronales", ("--- %s seconds ---" % (time.time() - start_time)))
        
    elif algoritmo == "rf":
        
        print ("inicio Random Forest")
        random_forest.predecir(data_x,data_y)
        print ("fin prediccion Random Forest", ("--- %s seconds ---" % (time.time() - start_time)))
