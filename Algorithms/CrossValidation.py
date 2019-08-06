from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

def nn_cv(hidden_layer_sizes, max_iter, data, targets):

    #Estimador de las Redes Neuronales
    estimator = MLPC(solver='sgd',learning_rate ='adaptive',
                     hidden_layer_sizes = (hidden_layer_sizes),
                     max_iter=max_iter, random_state=42)
    
    #Cross validation
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    
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
                    random_state=42, probability=True)
    
    #Cross validation
    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss', cv=4)
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
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features = max_features,
        random_state=42)
    
    #Cross validation
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()

def optimize_nn(data,targets):


    def nn_crossval(hidden_layer_sizes,max_iter):
        return nn_cv(int(hidden_layer_sizes), int(max_iter), data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=nn_crossval,
        pbounds={"hidden_layer_sizes": (4,2400),"max_iter":(100,1000)},
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=20)

    print("Final result:", optimizer.max)


def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=20)

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
            max_features = int(max_features),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (750, 2000),
            "min_samples_split": (45, 200),
            "max_features":(1,1200)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=20)

    print("Final result:", optimizer.max)
'''
if __name__ == "__main__":

    data, targets = #TODO colocar dataset_x y dataset_y

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
    optimize_rfc(data, targets)
'''
