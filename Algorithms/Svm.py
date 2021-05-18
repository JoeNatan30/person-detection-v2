#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:26:25 2018

@author: joe
"""

import joblib
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
#from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn import svm
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
#from sklearn.metrics import plot_confusion_matrix

def inicio_ejemplo(var):

    if var == 1:
        X = np.array([[1, 2, 2, 3],
                      [5, 2, 8, 5],
                      [2, 4, 7, 1.8],
                      [8, 2, 3, 8],
                      [1, 3, 1, 0.6],
                      [9, 1, 2, 11]])

        y = [1, 2, 1, 2, 0, 1]

    #    plt.scatter(X,y)
    #    plt.show()

        clf = svm.SVC(kernel='rbf',verbose = 1)
        modelo = clf.fit(X, y)

        modelo_filename = 'modelo_prueba.pkl'
        joblib.dump(modelo, modelo_filename)

    if(var==2):
        print ("modelo pkl")
        modelo_filename = 'modelo_prueba.pkl'
        clf = joblib.load(modelo_filename)

        print (clf.predict([[2.0, 3, 2.0, 3],
                           [0.58, 3, 0.76, 6.],
                           [10.58, 6, 10.76, 3]]))

    #a = -w[0] / w[1]

    #xx = np.linspace(0,12)
    #yy = a * xx - clf.intercept_[0] / w[1]

    #h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

    #plt.scatter(X[:, 0], X[:, 1], c = y)
    #plt.legend()
    #plt.show()

def entrenar(X, y): #dataset_train, resultados, dataset_realdataset_real

    expC = 30
    expGamma = 2

    svm_rbf = OneVsRestClassifier(svm.SVC(kernel='rbf', verbose=1, probability=True,
                                          C=2**expC, gamma=2**expGamma),
                                  n_jobs=-1)

    modelo = svm_rbf.fit(X, y)

    modelo_filename = './Results/svm_final_overfit.pkl'
    joblib.dump(modelo, modelo_filename)

def predecir(X_test, y_test):
    modelo_filename = './Results/svm_final_overfit.pkl'
    # modelo_filename = './Results/svm_final_overfit.pkl'
    svm_rbf = joblib.load(modelo_filename)

    y_pred = svm_rbf.predict(X_test)

    y_score = svm_rbf.predict_proba(X_test)

    y_true = y_test

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(threshold=np.inf)

    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    n_classes = 4
    class_names = ("cabeza", "torso", "brazos", "piernas")
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8,5.5))
    # precision recall curve
    precision = dict()
    recall = dict()
    plt.figure(figsize=(8,5.5))
    average_precision = dict()

    for i in range(n_classes):
        # ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Precition recall
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

        plt.plot(recall[i], precision[i], lw=2, label='Curva presición-recuerdo de la clase {0} (área = {1:0.2f})'
                                       ''.format(class_names[i], average_precision[i]))

    plt.xlabel("Recuerdo")
    plt.ylabel("precisión")
    plt.legend(loc="best")
    plt.title("SVM\nCurva de precisión vs recuerdo")
    plt.show()

    plt.figure(figsize=(8,5.5))
    plot_confusion_matrix(svm_rbf, X_test, y_true,
                          cmap=plt.cm.Blues,
                          display_labels=class_names)

    plt.title("SVM\nMatriz de confusión")
    plt.ylabel("Categoría verdadera")
    plt.xlabel("Categoría predecida")
    plt.show()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    plt.figure(figsize=(8,5.5))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Curva ROC de la clase {0} (área = {1:0.2f})'
                                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de falsos positivos')
    plt.ylabel('Ratio de verdaderos positivos')
    plt.title('SVM\nCurva ROC')
    plt.legend(loc="best")
    plt.show()
