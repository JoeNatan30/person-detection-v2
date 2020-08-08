
from sklearn.externals import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.multiclass import OneVsOneClassifier
from pandas_ml import ConfusionMatrix
import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")
#from sklearn_evaluation import plot

def inicio_ejemplo(var):
    X = np.array([[1, 2, 2, 3], 
                  [5, 2, 8, 5],
                  [2, 4, 7, 1.8],
                  [8, 2, 3, 8],
                  [1, 3, 1, 0.6],
                  [9, 1, 2, 11]])
            
    y = np.array([1,2,1,2,0,1])
    
    mpl = MLPClassifier(solver='adam')
    mpl.fit(X, y)
    print (mpl.predict([[2.0, 3, 2.0, 3],
                               [0.58, 3, 0.76, 6.],
                               [10.58, 6, 10.76, 3]]))


def entrenar(X, y):
    
    clf = OneVsOneClassifier(MLPClassifier(solver='adam',
                                           alpha=0.002342,
                                           beta_1=0.1, beta_2=0.1,
                                           verbose=1,
                                           hidden_layer_sizes = (20,663),
                                           max_iter=900),
                             n_jobs = -1)
    modelo = clf.fit(X, y)
    
    modelo_filename = 'nn_new_esPartePersona.pkl'
    #decision_tree_model_pkl = open(modelo_filename, 'wb')
    joblib.dump(modelo, modelo_filename)
    # Close the pickle instances
    #decision_tree_model_pkl.close()
    
    
def predecir(X_test, y_test):

    modelo_filename = 'nn_new_esPartePersona.pkl'
    #modelo = open(modelo_filename, 'rb')
    mpl = joblib.load(modelo_filename)
    
    y_pred = mpl.predict(X_test)
    
    #y_score = mpl.predict_proba(X_test)
    
    y_true = y_test
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(threshold=np.inf)
    
    cm = ConfusionMatrix(y_test, y_pred)
    cm.stats()
    cm.print_stats()
    '''
    preci = precision_score(y_true, y_pred, average='micro')
    conf = confusion_matrix(y_true, y_pred)
    aux = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    print "precision ", preci
    print "confusion ", conf
    print "Aux ", aux
    print "fpr ", fpr
    print "tpr ", tpr
    print "thresholds", thresholds
    '''
    
    #plot.precision_recall(y_true, y_score)
    
    #plot.confusion_matrix(y_true, y_pred)
    #print
    #plot.precision_recall(y_true, y_score)
    #print
    #plot.roc(y_true, y_score)
    
    #fpr, tpr, _ = roc_curve(y_test,  y_score) 
    #auc = roc_auc_score(y_test, y_score)   
    #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))   
    #plt.legend(loc=4)
    #plt.show()
    #'''
