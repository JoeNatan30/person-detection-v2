# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 18:17:12 2021

@author: Joe
"""

import json
import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def name():
    dtp = pd.read_pickle("./../datos/dataset/dataTrainX.pkl")
    dataset_x = dtp['dataTrainX']

    print(len(dataset_x[0])/3200)
###############################################################################


def showTricontour(results, x, y, xlim, ylim, nameX, nameY, modelName):

    fig, ax = plt.subplots(1, 1)

    ax.tricontour(x, y, results, levels=14,
                  linewidths=0.5, colors='k')
    cntr = ax.tricontourf(x, y, results,
                          levels=14, cmap="RdBu_r")

    fig.colorbar(cntr, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_title("Estimador F1 (%s)" % modelName)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
###############################################################################


def randomForestInfo():

    data = []

    for line in open('./results/rf_ovr_f1_final.json', 'r'):
        data.append(json.loads(line))

    results = []
    n_estimators = []
    max_features = []
    min_samples_split = []

    for rslt in data:

        for params in rslt['params'].items():
            if(params[0] == 'n_estimators'):
                x = params[1]
                n_estimators.append(params[1])
                neL = (50, 1000)

            if(params[0] == 'min_samples_split'):
                z = params[1]
                min_samples_split.append(params[1])
                mssL = (2, 150)

            if(params[0] == 'max_features'):
                y = params[1]
                #max_features.append(1/(pow(10, params[1])))
                max_features.append(params[1])
                mfL = (0, 3)

        results.append(rslt['target'])

#    showTricontour(results, n_estimators, max_features, neL, mfL, "Estimadores", "Características máximas")
    showTricontour(results, n_estimators, min_samples_split,
                   neL, mssL,
                   "N° de estimadores", "División mímima de la muestra",
                   "Bosque aleatorios")
#    showTricontour(results, min_samples_split, max_features, mssL, mfL, "División mímima de la muestra", "Características máximas")


def SVMInfo():

    data = []

    for line in open('./results/f1_macro/svc_f1_macro.json', 'r'):
        data.append(json.loads(line))

    results = []
    expC = []
    expGamma = []

    for rslt in data:

        for params in rslt['params'].items():
            if(params[0] == 'expC'):
                x = 2 ** params[1]
                expC.append(params[1])
                ecL = (-5, 30)

            if(params[0] == 'expGamma'):
                y = 2 ** params[1]
                expGamma.append(params[1])
                egL = (-15, 4)

        results.append(rslt['target'])

    showTricontour(results, expC, expGamma,
                   ecL, egL, 
                   "exp C", "exp Gamma", 
                   "SVM")


def NeuralNetworkInfo():
    
    data = []

    for line in open('./results/f1_macro/nn_f1_macro_log.json', 'r'):
        data.append(json.loads(line))

    results = []
    hidden_layer_sizes_1 = []
    hidden_layer_sizes_2 = []
    for rslt in data:

        for params in rslt['params'].items():

            if(params[0] == 'hidden_layer_sizes_1'):
                d = params[1]
                hidden_layer_sizes_1.append(params[1])
                hls1L = (20, 716)

            if(params[0] == 'hidden_layer_sizes_2'):
                e = params[1]
                hidden_layer_sizes_2.append(params[1])
                hls2L = (20, 715)

        results.append(rslt['target'])

    showTricontour(results, hidden_layer_sizes_1, hidden_layer_sizes_2,
                   hls1L, hls2L,
                   "N° de neuronas escondidas 1", "N° de neuronas escondidas 2",
                   "Redes neuronales")
    
SVMInfo()
#NeuralNetworkInfo()
#randomForestInfo()
