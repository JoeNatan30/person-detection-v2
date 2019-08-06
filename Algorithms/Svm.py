#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:26:25 2018

@author: joe
"""
	
from sklearn.externals import joblib
import numpy as np
from pandas_ml import ConfusionMatrix
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")
#from sklearn_evaluation import plot
from sklearn import svm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def inicio_ejemplo(var):
    
    if var == 1:
        X = np.array([[1,2, 2,3],
                      [5,2, 8,5],
                      [2,4, 7,1.8],
                      [8,2, 3,8],
                      [1,3, 1,0.6],
                      [9,1, 2,11]])
        
        y = [1,2,1,2,0,1]
        
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
        
    svm_rbf = svm.SVC(kernel='rbf',verbose = 1,C=0.1,gamma=0.1)
    svm_rbf.probability = True

    modelo = svm_rbf.fit(X, y)
    
    modelo_filename = 'svm_new_esPartePersona.pkl'
    joblib.dump(modelo, modelo_filename)

def predecir(X_test, y_test):

    modelo_filename = 'svm_new_esPartePersona.pkl'
    svm_rbf = joblib.load(modelo_filename)
    
    y_pred = svm_rbf.predict(X_test)
    
    y_score = svm_rbf.predict_proba(X_test)
    y_true = y_test
    
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    np.set_printoptions(threshold=np.inf)
    
    cm = ConfusionMatrix(y_test, y_pred)
    cm.stats()
    cm.print_stats()

    #plot.precision_recall(y_true, y_score)
    #plot.roc(y_true, y_score)
    #plot.confusion_matrix(y_true, y_pred)
    
    '''
    fpr, tpr, _ = roc_curve(y_test,  y_score) 
    auc = roc_auc_score(y_test, y_score)   
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))   
    plt.legend(loc=4)
    plt.show()
    '''
    print ("predecir")

#inicio_ejemplo(2)

