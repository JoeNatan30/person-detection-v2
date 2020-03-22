import pandas as pd
import numpy as np

def entrenamiento(cant_PCD,porcentaje):
    
    #Algoritmos
    dataset_x = []
    dataset_y = []
    
    tope = cant_PCD * porcentaje / 100
    
    for pos in range(cant_PCD):
    
        if(pos < tope):
            print (pos)
            #Toma de datos procesados
            dtp = pd.read_pickle("./../datos/procesado/procesado_%d.pkl"%(pos))
        
            #Longitud del arreglo tomado
            longit = len(dtp['es'])

            #Para guardar cada arreglo por unidad
            for ind in range(longit):
                
                dataset_x.append(dtp['histograma'][ind])
                dataset_y.append(dtp['es'][ind])

    #conversion a numpy array
    dataset_X_arr = np.array(dataset_x)
    dataset_Y_arr = np.array(dataset_y)
    
    return dataset_X_arr, dataset_Y_arr

def prueba(cant_PCD,porcentaje):
    
    base = cant_PCD * porcentaje / 100
    base -= 1
    
    #Algoritmos
    dataset_x = []
    dataset_y = []

    for pos in range(cant_PCD):
  
        if(pos > base):
            
            #Toma de datos procesados
            dtp = pd.read_pickle('./../datos/procesado/procesado_%d.pkl'%(pos))
        
            #Longitud del arreglo tomado
            longit = len(dtp['es'])

            #Para guardar cada arreglo por unidad
            for ind in range(longit):
             
                dataset_x.append(dtp['histograma'][ind])
                dataset_y.append(dtp['es'][ind])

    #conversion a numpy array
    dataset_X_arr = np.array(dataset_x)
    dataset_Y_arr = np.array(dataset_y)
    
    return dataset_X_arr, dataset_Y_arr