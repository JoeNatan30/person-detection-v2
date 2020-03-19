#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:58:23 2019

@author: joe
"""
import procesos

def main(proceso,tipo):
    
    cant_PCD = 4000           # numero total de archivos de datos sin procesar
    porcentaje = 75         # porcentaje para el training
    #tamano_conjunto = 300   # tamano del conjunto de puntos a tomar
    tamano_conjunto = 800   # tamano del conjunto de puntos a tomar
    algoritmo = "svm"       # "svm" "rf" "rn"
    version = 1             # version del kinect
    max_proces_paral = 4    # Maximo de procesos paralelos (se cuenta desde cero)
    pos_proces_paral = 0    # posicion de procesos paralelos (debe inicar en cero)
    #rangeOfDiff = 0.0000000149     # Range of Difference used during normal calculation
    rangeOfDiff = 0.000000014889
    normalPresicion = 7
    verbose = True          # to have more detail of the process
    
    if proceso == "captura":
        procesos.lectura_(cant_PCD)
    
    elif proceso == "procesamiento_total":
        procesos.procesamiento_train(cant_PCD,porcentaje,tamano_conjunto,
                                     version,max_proces_paral,
                                     pos_proces_paral,rangeOfDiff, verbose)
        
        procesos.procesamiento_real(cant_PCD,porcentaje,tamano_conjunto,
                                    version,max_proces_paral,
                                    pos_proces_paral,rangeOfDiff, verbose)

    elif proceso == "procesamiento_train":
        procesos.procesamiento_train(cant_PCD,porcentaje,tamano_conjunto,
                                     version,max_proces_paral,
                                     pos_proces_paral,rangeOfDiff, verbose)
        
    elif proceso == "procesamiento_real":
        procesos.procesamiento_real(cant_PCD,porcentaje,tamano_conjunto,
                                    version,max_proces_paral,
                                    pos_proces_paral,rangeOfDiff, verbose)

    elif proceso == "entrenamiento":
        procesos.entrenamiento(cant_PCD,porcentaje,algoritmo,verbose)
        
    elif proceso == "validacion_cruzada":
        procesos.validacion_cruzada(cant_PCD,porcentaje,algoritmo,verbose)
    
    elif proceso == "prueba":
        procesos.prueba(cant_PCD,porcentaje,algoritmo,verbose)
        
    elif proceso == "medicion":
        procesos.medition(cant_PCD,tamano_conjunto,
                          version,max_proces_paral,
                          pos_proces_paral,rangeOfDiff,
                          tipo,normalPresicion ,verbose)

    else:
        print ("opciones:")
        print ("captura -> Solo captura desde kinet v2")
        print ("procesamiento_total" )
        print ("procesamiento_train")
        print ("procesamiento_real")
        print ("entrenamiento")
        print ("validacion_cruzada")
        print ("prueba")
        print ("medicion:")
        print ("    ruido-rangeOfDiff")
        print ("    ruido-rangeOfSamples")
        print ("    ruido-normalPrecision")

         
###############################################################################
# MAIN
###############################################################################
"""
 opciones = "captura" -> Solo captura desde kinet v2
            "procesamiento_total" 
            "procesamiento_train"
            "procesamiento_real"
            "entrenamiento"
            "validacion_cruzada"
            "prueba"
"""
#main("medicion","ruido-normalPrecision")
main("procesamiento_train","")
