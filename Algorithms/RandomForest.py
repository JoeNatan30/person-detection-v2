#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:38:14 2018

@author: joe
"""
from sklearn.externals import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import precision_recall_fscore_support as score
#import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn_evaluation import plot


def inicio_ejemplo(var):
    X = np.array([[1, 2, 2, 3], 
                  [5, 2, 8, 5],
                  [2, 4, 7, 1.8],
                  [8, 2, 3, 8],
                  [1, 3, 1, 0.6],
                  [9, 1, 2, 11]])
            
    y = np.array([1,2,1,2,0,1])
    
    rf = RandomForestClassifier(n_estimators= 100)
    rf.fit(X, y)
    
    print (rf.predict([[2.0, 3, 2.0, 3],
                      [0.58, 3, 0.76, 6.],
                      [10.58, 6, 10.76, 3]]))


def entrenar(X, y):
    
    rf = RandomForestClassifier(n_estimators=1833, min_samples_split=54,verbose=1)
    modelo = rf.fit(X, y)
    
    modelo_filename = 'rf_new_esPartePersona.pkl'
    #decision_tree_model_pkl = open(modelo_filename, 'wb')
    joblib.dump(modelo, modelo_filename)
    # Close the pickle instances
    #decision_tree_model_pkl.close()
    


def predecir(X_test, y_test):

    #print "modelo pkl"
    modelo_filename = 'rf_new_esPartePersona.pkl'
    #modelo = open(modelo_filename, 'rb')
    svm_rbf = joblib.load(modelo_filename)
    
    y_pred = svm_rbf.predict(X_test)
    
    y_score = svm_rbf.predict_proba(X_test)
        
    y_true = y_test
    
    #print y_pred
    #print y_test
    #print y_score

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(threshold=np.inf)

    cm = ConfusionMatrix(y_test, y_pred)
    cm.print_stats()
    cm.stats()
    '''
    arr = [0,0,0,0] 
    matrix = []
    matrix.append(arr)
    matrix.append(arr)
    matrix.append(arr)
    matrix.append(arr)
    
    
    print y_true
    print
    print y_pred
    
    for true in y_true:
        
        for pred in y_pred:
            
            matrix[true][pred] += 1

    print matrix
    '''    
    
    #precision, recall, fscore, support = score(y_test, y_pred)
    #print('precision: {}'.format(precision))
    #print('recall: {}'.format(recall))
    #print('fscore: {}'.format(fscore))
    #print('support: {}'.format(support))
    #preci = precision_score(y_true, y_pred, average='weighted')
    #conf = confusion_matrix(y_true, y_pred)
    #aux = roc_auc_score(y_true, y_score,average='weighted')
    #fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    #print "precision ", preci
    #print "confusion ", conf
    #print "Aux ", aux
    #print "fpr ", fpr
    #print "tpr ", tpr
    #print "thresholds", thresholds
    #plt.figure()
    #plot.precision_recall(y_true, y_score)
    #plot.roc(y_true, y_score)
    plot.confusion_matrix(y_true, y_pred)
    
    '''
    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    plot.confusion_matrix(cnf_matrix, classes= clases)
    
    plot.precision_recall(y_true, y_score)
    plot.roc(y_true, y_score)
    plot.confusion_matrix(y_true, y_pred)
    '''
    '''
    fpr, tpr, _ = roc_curve(y_test,  y_score) 
    auc = roc_auc_score(y_test, y_score)   
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))   
    plt.legend(loc=4)
    plt.show()
    '''
