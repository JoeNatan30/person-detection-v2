import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd


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
    
    clf = OneVsRestClassifier(MLPClassifier(solver='adam',
                                           alpha=0.002342,
                                           beta_1=0.9, beta_2=0.999,
                                           verbose=1,
                                           hidden_layer_sizes = (20,663),
                                           max_iter=900),
                             n_jobs = -1)
    modelo = clf.fit(X, y)
    
    modelo_filename = 'nn_ovr_f1.pkl'
    #decision_tree_model_pkl = open(modelo_filename, 'wb')
    joblib.dump(modelo, modelo_filename)
    # Close the pickle instances
    #decision_tree_model_pkl.close()
    
    
def predecir(X_test, y_test):

    modelo_filename = 'nn_ovo_f1.pkl'
    #modelo = open(modelo_filename, 'rb')
    mpl = joblib.load(modelo_filename)
    
    y_pred = mpl.predict(X_test)
    
    y_score = mpl.predict_proba(X_test)
    
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

    plot_confusion_matrix(mpl, X_test, y_true,
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