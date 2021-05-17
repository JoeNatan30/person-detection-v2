import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def entrenamiento(cant_PCD,porcentaje):

    #Algoritmos
    dataset_x = []
    dataset_y = []
    
    dtp = pd.read_pickle("./dataset/dataTrainX.pkl")
    dataset_x = dtp['dataTrainX']
 
    dtp = pd.read_pickle("./dataset/dataTrainY.pkl")
    dataset_y = dtp['dataTrainY']

    #conversion a numpy array
    dataset_X_arr = np.array(dataset_x[0])
    dataset_Y_arr = np.array(dataset_y)

    total = cant_PCD * 32

    return dataset_X_arr[:total], dataset_Y_arr[:total]


def prueba(cant_PCD, porcentaje):

    # Algoritmos
    dataset_x = []
    dataset_y = []

    dtp = pd.read_pickle("./dataset/dataTestX.pkl")
    dataset_x = dtp['dataTestX']

    dtp = pd.read_pickle("./dataset/dataTestY.pkl")
    dataset_y = dtp['dataTestY']

    # conversion a numpy array
    dataset_X_arr = np.array(dataset_x[0])
    dataset_Y_arr = np.array(dataset_y)

    return dataset_X_arr, dataset_Y_arr


def dataset(cantPcd, pivot, verbose):

    arrTrain = np.asarray(range(pivot))
    arrReal = np.asarray(range(pivot, cantPcd))

    x = []
    y = []

    for posTrain in arrTrain:

        dtp = pd.read_pickle('./../datos/procesado/procesado_%d.pkl' %
                             (posTrain))

        longit = len(dtp['es'])

        # Para guardar cada arreglo por unidad
        for ind in range(longit):

            x.append(dtp['histograma'][ind])
            y.append(dtp['es'][ind])

    pivot = len(y)

    for posTest in arrReal:

        dtp = pd.read_pickle('./../datos/procesado/procesado_%d.pkl' %
                             (posTest))

        longit = len(dtp['es'])

        # Para guardar cada arreglo por unidad
        for ind in range(longit):

            x.append(dtp['histograma'][ind])
            y.append(dtp['es'][ind])

    pivot_2 = len(y)

    if(verbose):
        print("Data Procesada complete")

    for posReal in arrReal:

        dtp = pd.read_pickle('./../datos/real/real%d.pkl' %
                             (posReal))

        longit = len(dtp['histograma'])

        # Para guardar cada arreglo por unidad
        for ind in range(longit):

            x.append(dtp['histograma'][ind])

    if(verbose):
        print("Data Real complete")

    dataset_x, norms = normalize(x, return_norm=True, axis=0)

    dataTrainX = {'dataTrainX': [dataset_x[:pivot]]}
    dtpTrainX = pd.DataFrame(dataTrainX)
    dtpTrainX.to_pickle('./../datos/dataset/dataTrainX.pkl')

    dataTestX = {'dataTestX': [dataset_x[pivot:pivot_2]]}
    dtpTestX = pd.DataFrame(dataTestX)
    dtpTestX.to_pickle('./../datos/dataset/dataTestX.pkl')

    dataReal = {'dataReal': [dataset_x[pivot_2:]]}
    dptReal = pd.DataFrame(dataReal)
    dptReal.to_pickle('./../datos/dataset/dataReal.pkl')

    dataTrainY = {'dataTrainY': y[:pivot]}
    dtpTrainY = pd.DataFrame(dataTrainY)
    dtpTrainY.to_pickle('./../datos/dataset/dataTrainY.pkl')

    dataTestY = {'dataTestY': y[pivot:]}
    dtpTestY = pd.DataFrame(dataTestY)
    dtpTestY.to_pickle('./../datos/dataset/dataTestY.pkl')

    dataNorms = {'norms': norms}
    dptNorms = pd.DataFrame(dataNorms)
    dptNorms.to_pickle('./../datos/dataset/dataNorms.pkl')

    if(verbose):
        print("Data Normalized")
