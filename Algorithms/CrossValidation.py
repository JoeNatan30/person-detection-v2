from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,  average_precision_score
from sklearn import config_context

def nn_cv(hidden_layer_sizes_1, hidden_layer_sizes_2, alpha, beta_1, beta_2, data, targets):
    """Neural network validation
    """
    #Estimador de las Redes Neuronales
    estimator = OneVsOneClassifier(MLPC(solver='adam',
                     alpha = alpha,
                     beta_1 = beta_1,
                     beta_2 = beta_2,
                     hidden_layer_sizes = (hidden_layer_sizes_1 ,hidden_layer_sizes_2),
                     max_iter = 900, random_state=42), n_jobs=-1)

    skf = StratifiedKFold(n_splits=4)

    def getScores(estimator, x, y):
        yPred = estimator.predict(x)
        labels = np.unique(yPred)

        return (accuracy_score(y,yPred),
                precision_score(y, yPred, average='macro', labels=labels),
                recall_score(y, yPred, average='macro', labels=labels))

    def my_scorer(estimator, x,y):
       a, p, r = getScores(estimator, x, y)
       return (a+p+r)/3

    #Cross validation
    cval = cross_val_score(estimator, data, targets,n_jobs=-1,
                           scoring = my_scorer, cv=skf)
    
    return cval.mean()

def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    #Estimador del SVM
    estimator = SVC(kernel='rbf', C=C, gamma=gamma,
                    random_state=42)

    skf = StratifiedKFold(n_splits=4)

    def getScores(estimator, x, y):
        yPred = estimator.predict(x)
        labels = np.unique(yPred)
        return (accuracy_score(y, yPred),
            precision_score(y, yPred, average = 'macro', labels=labels),
            recall_score(y, yPred, average ='macro',labels=labels))

    def my_scorer(estimator, x, y):
        a, p, r = getScores(estimator, x, y)
        return (a+p+r)/3

    #Cross validation
    cval = cross_val_score(estimator, data, targets, n_jobs=-1, scoring=my_scorer, cv=skf)
    return cval.mean()


def rfc_cv(n_estimators, min_samples_split,max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    
    #Estimador del Random Forest
    estimator = OneVsRestClassifier(RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features = max_features,
        n_jobs = -1,
        random_state=42),n_jobs=-1)

    def getScores(estimator, x, y):
        yPred = estimator.predict(x)
        labels = np.unique(yPred)
        return (accuracy_score(y, yPred),
                precision_score(y, yPred, average='macro', labels= labels),
                recall_score(y, yPred, average='macro', labels= labels))

    def my_scorer(estimator, x, y):
        a, p, r = getScores(estimator, x, y)
        return (a+p+r)/3

    skf = StratifiedKFold(n_splits=4)

    #Cross validation
    cval = cross_val_score(estimator, data, targets, scoring=my_scorer, cv=skf,n_jobs=-1)
    return cval.mean()

def optimize_nn(data,targets):


    def nn_crossval(hidden_layer_sizes_1, hidden_layer_sizes_2, alpha, beta_1, beta_2):
        return nn_cv(int(hidden_layer_sizes_1),
                     int(hidden_layer_sizes_2),
                     alpha = alpha,
                     beta_1 = 0.1,
                     beta_2 = 0.1,
                     data=data, targets=targets)

    optimizer = BayesianOptimization(
            f=nn_crossval,
            pbounds={
             "hidden_layer_sizes_1": (20,1000),
             "hidden_layer_sizes_2":(20,1000),
             "alpha":(0.00001,0.1),
             "beta_1":(0.1,0.9),
             "beta_2":(0.1,0.9)},
            random_state=42,
            verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 2 ** expC
        gamma = 2 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-5, 3), "expGamma": (-15, 3)},
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split,max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """

        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features = max_features,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (200, 1000),
            "min_samples_split": (2, 200),
            "max_features":(0.0000001,1)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
'''
if __name__ == "__main__":

    data, targets = #TODO colocar dataset_x y dataset_y

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
    optimize_rfc(data, targets)
'''
