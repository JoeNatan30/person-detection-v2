#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:58:23 2019

@author: joe
"""
import procesos

def main(proceso):
    
    cant_PCD = 55           # numero total de archivos de datos sin procesar
    porcentaje = 75         # porcentaje para el training
    tamano_conjunto = 300   # tamano del conjunto de puntos a tomar
    algoritmo = "svm"       # "svm" "rf" "rn"
    version = 1             # version del kinect
    max_proces_paral = 4    # Maximo de procesos paralelos (se cuenta desde cero)
    pos_proces_paral = 0    # posicion de procesos paralelos (debe inicar en cero)
    
    if proceso == "captura":
        procesos.lectura_(cant_PCD)
    
    elif proceso == "procesamiento_total":
        procesos.procesamiento_train(cant_PCD,porcentaje,tamano_conjunto,version,max_proces_paral,pos_proces_paral)
        procesos.procesamiento_real(cant_PCD,porcentaje,tamano_conjunto,version,max_proces_paral,pos_proces_paral)

    elif proceso == "procesamiento_train":
        procesos.procesamiento_train(cant_PCD,porcentaje,tamano_conjunto,version,max_proces_paral,pos_proces_paral)
        
    elif proceso == "procesamiento_real":
        procesos.procesamiento_real(cant_PCD,porcentaje,tamano_conjunto,version,max_proces_paral,pos_proces_paral)

    elif proceso == "entrenamiento":
        procesos.entrenamiento(cant_PCD,porcentaje,algoritmo)
        
    elif proceso == "validacion_cruzada":
        procesos.validacion_cruzada(cant_PCD,porcentaje,algoritmo)
    
    elif proceso == "prueba":
        procesos.prueba(cant_PCD,porcentaje,algoritmo)

    else:
        print ("opciones:")
        print ("captura -> Solo captura desde kinet v2")
        print ("procesamiento_total" )
        print ("procesamiento_train")
        print ("procesamiento_real")
        print ("entrenamiento")
        print ("validacion_cruzada")
        print ("prueba")
        
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
main("procesamiento_train")