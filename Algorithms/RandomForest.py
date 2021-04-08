# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:38:14 2018

@author: joe
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")

# from sklearn_evaluation import plot


def inicio_ejemplo(var):
    X = np.array([[1, 2, 2, 3],
                  [5, 2, 8, 5],
                  [2, 4, 7, 1.8],
                  [8, 2, 3, 8],
                  [1, 3, 1, 0.6],
                  [9, 1, 2, 11]])

    y = np.array([1, 2, 1, 2, 0, 1])

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    print(rf.predict([[2.0, 3, 2.0, 3],
                      [0.58, 3, 0.76, 6.],
                      [10.58, 6, 10.76, 3]]))


def entrenar(X, y):
    
    max_features = 1/(pow(10,0.5573003895001497))

    # F1
    rf = RandomForestClassifier(
                             n_estimators=int(583.3696128501642),
                             min_samples_split=int(4.041961432004625),
                             max_features=max_features,
                             verbose=1,
                             n_jobs=-1)
    '''
    # logloss
    rf = OneVsRestClassifier(RandomForestClassifier(
                             n_estimators=int(603.6),
                             min_samples_split=int(82.67),
                             max_features=0.23920097566199192,
                             verbose=1,
                             n_jobs=-1), n_jobs=-1)
    '''
    modelo = rf.fit(X, y)

    modelo_filename = './Results/randomForest/f1/rf_ovr_f1_2.pkl'

    joblib.dump(modelo, modelo_filename)
    # Close the pickle instances


def predecir(X_test, y_test):

    modelo_filename = './Results/randomForest/f1/rf_ovr_f1_2.pkl'

    rf_model = joblib.load(modelo_filename)

    y_pred = rf_model.predict(X_test)

    y_score = rf_model.predict_proba(X_test)

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

    # precision recall curve
    precision = dict()
    recall = dict()

    for i in range(n_classes):
        # ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Precition recall
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])

        plt.plot(recall[i], precision[i], lw=2, label=class_names[i])

    plt.xlabel("Recuerdo")
    plt.ylabel("precisión")
    plt.legend(loc="best")
    plt.title("Curva de precisión vs recuerdo")
    plt.show()

    plot_confusion_matrix(rf_model, X_test, y_true,
                          cmap=plt.cm.Blues,
                          display_labels=class_names)

    plt.show()
    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='Área bajo la curva: %0.2f)' %
                 roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC: %s' % class_names[i])
        plt.legend(loc="lower right")
        plt.show()

    print(average_precision_score(y_test, y_pred, average='macro'))
