# from lectura import lectura_de_kinect as lk
# from lectura import lectura_xml
from NoiseReduction import ReduceNoise, KdtreeStructure, ReduceNoiseUtils
from Segmentation import RansacAlgorithm
from Characterization import Fpfh, Descriptor
from Characterization import Procesados
from Algorithms import CrossValidation
from Algorithms import Svm, NeuralNetwork, RandomForest
from open3d import io

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Definitions    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###############################################################################
'''
def lectura_(cant_PCD):
    lk.iniciar(cant_PCD)
'''


###############################################################################
def segmentation(pos, pc, kdtree, v, verbose):

    if(verbose):
        print("RansacAlgorithm.iniciar")

    pc_sin_seg, kdtree_sin_seg = RansacAlgorithm.iniciar(pc, kdtree, v)

    io.write_point_cloud(
        str.encode('./../datos/segmentado/segmentado_%d.pcd' % (pos)),
        pc_sin_seg)

    return pc_sin_seg, kdtree_sin_seg


###############################################################################
def histograma(pc, kdtree, tamano, verbose):

    if(verbose):
        print("Fpfh.inicio")
    list_fpfh_point = Fpfh.inicio(pc, kdtree, tamano, verbose)

    return list_fpfh_point


###############################################################################
def descripcion(fpfh, tamano, pcdSize, pc, verbose):

    if(verbose):
        print("Descriptor.inicio")
    conj_fpfh, conj_ind, conj_extr = Descriptor.inicio(fpfh, pc,
                                                       tamano, pcdSize,
                                                       verbose)

    return conj_fpfh, conj_ind, conj_extr


###############################################################################
def descripcion_train(pos, tamano, fpfh, pc, kdtree, verbose):

    if(verbose):
        print("Descriptor.obtener_descriptores_train")

    dataset_X, dataset_Y, conj_extre = Descriptor.obtener_descriptores_train(
        pos, fpfh, pc, kdtree, tamano)

    return dataset_X, dataset_Y, conj_extre


def plotBodyParts(dataset_X, dataset_Y):

    cero = []
    uno = []
    dos = []
    tres = []

    for pos in range(16):
        if(dataset_Y[pos] == 0):
            cero.append(dataset_X[pos])
        if(dataset_Y[pos] == 1):
            uno.append(dataset_X[pos])
        if(dataset_Y[pos] == 2):
            dos.append(dataset_X[pos])
        if(dataset_Y[pos] == 3):
            tres.append(dataset_X[pos])

        for datito in cero:
            plt.plot(datito)
        plt.show()
        for datito in uno:
            plt.plot(datito)
        plt.show()
        for datito in dos:
            plt.plot(datito)
        plt.show()
        for datito in tres:
            plt.plot(datito)
        plt.show()


def guardar_datos_reales(pos, data_x):

    datos = {'histograma': data_x}

    dtp = pd.DataFrame(datos)

    dtp.to_pickle('./../datos/real/real%d.pkl' % pos)


###############################################################################
def guardar_datos_procesados(pos, data_x, data_y):
    datos = {'es': data_y,
             'histograma': data_x}

    dtp = pd.DataFrame(datos)

    dtp.to_pickle('./../datos/procesado/procesado_%d.pkl' % pos)

    # para leer el fpfh de un archivo
    # fpfh = pd.read_pickle('data/entrenamiento/fpfh_%d.pkl'%pos)
    # fpfh_list = fpfh.values


###############################################################################
def procesamiento_train(cant_PCD, porcentaje, tamano, version,
                        max_paral, pos_paral, rangeOfDiff, verbose):

    rest = pos_paral

    tope = cant_PCD * porcentaje / 100

    for pos in range(cant_PCD):
        # print(pos, cant_PCD, porcentaje)
        if(rest == pos % max_paral < tope):

            print("posicion: ", pos)
            print("menor que: ", rangeOfDiff)
            '''
            # Ruido
            if (verbose):
                print("///////////////////////\nReduceNoise.ruido")

            pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                    rangeOfDiff,
                    pos,
                    verbose)

            # segmentacion
            if(verbose):
                print("///////////////////////\nsegmentation")
            path = './../datos/sin_ruido/sin_ruido_%d.pcd' % pos
            pc_sin_ruido, kdtree_sin_ruido, size = KdtreeStructure.getKdtreeFromPointCloudDir(path)
            pc_seg, kdtree_seg = segmentation(pos, pc_sin_ruido,
                                              kdtree_sin_ruido,
                                              version, verbose)
            writeDir = './../datos/segmentado/segmentado_%d.pcd' % pos
            ReduceNoiseUtils.saveFile(writeDir, pc_seg)
            #print(len(np.asarray(pc_seg.points)))
            '''
            # FPFH
            if (verbose):
                print("///////////////////////\nHistogram")

            path = './../datos/segmentado/segmentado_%d.pcd' % pos
            pc_seg, kdtree_seg, pcdSize = KdtreeStructure.getKdtreeFromPointCloudDir(path)

            fpfh_list = histograma(pc_seg, kdtree_seg, tamano, verbose)

            # Descriptor
            if (verbose):
                print("///////////////////////\nDescriptor")

            dataset_X, dataset_Y, conj_extre = descripcion_train(
                pos, tamano, pcdSize, fpfh_list, pc_seg, kdtree_seg, verbose)

            # Guardado
            if(verbose):
                print("Save Processed Data")

            guardar_datos_procesados(pos, dataset_X, dataset_Y)

            print("END - Process: ", pos)


###############################################################################
def procesamiento_real(cant_PCD, porcentaje, tamano, version,
                       max_paral, pos_paral, rangeOfDiff, verbose):

    if(verbose):
        print("///////////////////////Procesamiento real")

    rest = pos_paral

    base = cant_PCD * porcentaje / 100
    base -= 1

    for pos in range(cant_PCD):

        # if(rest == pos % max_paral and pos > base):
        if(pos > base):
            print("")
            print(pos)
            '''
            # Ruido
            if(verbose):
                print("///////////////////////\nReduce Noise")
            pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                    rangeOfDiff,
                    pos,
                    verbose)

            # segmentacion
            if(verbose):
                print("///////////////////////\nSegmentation")
            pc_seg, kdtree_seg = segmentation(pos, pc_sin_ruido,
                                              kdtree_sin_ruido,
                                              version, verbose)
            '''
            # FPFH
            if(verbose):
                print("///////////////////////\nHistogram")
            pc_seg, kdtree_seg, pcdSize = KdtreeStructure.getKdtreeFromPointCloudDir('./../datos/segmentado/segmentado_%d.pcd'%pos)
            fpfh_list = histograma(pc_seg, kdtree_seg, tamano, verbose)

            # Descriptor
            if(verbose):
                print("///////////////////////\nDescriptor")
            dataset_X, dataset_Y, conj_extre = descripcion(
                fpfh_list, tamano, pcdSize, pc_seg, verbose)

            # Guardado
            if(verbose):
                print("///////////////////////\nSave")
            guardar_datos_reales(pos, dataset_X)


def preparacion_dataset(cantPcd, porcentaje, verbose):

    pivot = int(cantPcd * porcentaje / 100)

    if(verbose):
        print("///////////////////////\ncreate dataset")
    Procesados.dataset(cantPcd, pivot, verbose)

##############################################################################
def entrenamiento(cant, porcentaje, algoritmo, verbose):

    data_x, data_y = Procesados.entrenamiento(cant, porcentaje)
    data_x = data_x
    start_time = time.time()

    if algoritmo == "svm":

        print("inicio entrenamiento SVM")
        Svm.entrenar(data_x, data_y)
        print("SVM entrenado", ("--- %s seconds ---" %
                                (time.time() - start_time)))

    elif algoritmo == "rn":

        print("inicio entrenamiento redes neuronales")
        NeuralNetwork.entrenar(data_x, data_y)
        print("Redes Neuronales entrenado", ("--- %s seconds ---" %
                                             (time.time() - start_time)))

    elif algoritmo == "rf":

        print("inicio Random Forest")
        RandomForest.entrenar(data_x, data_y)
        print("Random Forest entrenado", ("--- %s seconds ---" %
                                          (time.time() - start_time)))


###############################################################################
def validacion_cruzada(cant, porcentaje, algoritmo, verbose):
    print("cargar datos")
    data_x, data_y = Procesados.entrenamiento(cant, porcentaje)

    start_time = time.time()

    if algoritmo == "svm":

        print("inicio cross validation SVM")
        CrossValidation.optimize_svc(data_x, data_y)
        print("fin cross validation SVM", ("--- %s seconds ---" %
                                           (time.time() - start_time)))

    elif algoritmo == "rn":

        print("inicio cross validation Redes Neuronales")
        CrossValidation.optimize_nn(data_x, data_y)

        print("fin cross validation Redes Neuronales",
              ("--- %s seconds ---" % (time.time() - start_time)))

    elif algoritmo == "rf":

        print("inicio cross validation Random Forest")
        CrossValidation.optimize_rfc(data_x, data_y)
        print("fin cross validation Random Forest",
              ("--- %s seconds ---" % (time.time() - start_time)))


###############################################################################
def prueba(cant, porcentaje, algoritmo, verbose):

    data_x, data_y = Procesados.prueba(cant, porcentaje)

    start_time = time.time()

    if algoritmo == "svm":

        print("inicio prediccion SVM")
        Svm.predecir(data_x, data_y)
        print("fin prediccion SVM", ("--- %s seconds ---" %
                                     (time.time() - start_time)))

    elif algoritmo == "rn":

        print("inicio prediccion Redes Neuronales")
        NeuralNetwork.predecir(data_x, data_y)
        print("fin prediccion Redes Neuronales", ("--- %s seconds ---"
                                                  % (time.time() -
                                                     start_time)))

    elif algoritmo == "rf":

        print("inicio Random Forest")
        RandomForest.predecir(data_x, data_y)
        print("fin prediccion Random Forest", ("--- %s seconds ---" %
                                               (time.time() - start_time)))
    

###############################################################################
def medition(cant_PCD, tamano_conjunto, version,
             max_paral, pos_paral, rangeOfDiff, tipo, precision, verbose):

    print("medicion de " + tipo)
    pos = 1000

    proceso = tipo.split("-")[0]

    if proceso == "ruido":
        ReduceNoise.medition(rangeOfDiff, pos, tipo, precision, verbose)

    if proceso == "ransac":

        contador = 0
        rangeList = range(351)
        rangeList = [340, 271]
        dictionary = {}

        for pos in rangeList:
            print("pos ======> ", pos)
            pc_sin_ruido, kdtree_sin_ruido = ReduceNoise.ruido(
                rangeOfDiff, int(pos), verbose)
            rslts = RansacAlgorithm.medition(
                pc_sin_ruido, kdtree_sin_ruido, verbose)

            contador += 1
            dictionary[pos] = rslts
            print("###### COUNTER: %d SAVED ######" % contador)

        df = pd.DataFrame.from_dict(dictionary)
        df.to_json('./medition_ransac_results_manual_last_mod.json')
