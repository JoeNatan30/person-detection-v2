from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold


# #############################################################################
# NEURAL NETWORK

def nn_cv(hidden_layer_sizes_1, hidden_layer_sizes_2,
          data, targets):

    # Estimador de las Redes Neuronales
    estimator = OneVsRestClassifier(MLPC(solver='adam',
                                         hidden_layer_sizes=(
                                             hidden_layer_sizes_1,
                                             hidden_layer_sizes_2),
                                         random_state=41),
                                    n_jobs=-1)

    skf = StratifiedKFold(n_splits=4)

    # Cross validation
    cval = cross_val_score(estimator, data, targets, n_jobs=-1,
                           scoring='f1_macro', cv=skf)

    return cval.mean()


def optimize_nn(data, targets):

    def nn_crossval(hidden_layer_sizes_1, hidden_layer_sizes_2):

        return nn_cv(int(hidden_layer_sizes_1),
                     int(hidden_layer_sizes_2),
                     data=data, targets=targets)

    optimizer = BayesianOptimization(
            f=nn_crossval,
            pbounds={
             "hidden_layer_sizes_1": (20, 800),
             "hidden_layer_sizes_2": (20, 800)},
            random_state=41,
            verbose=2
    )

    optimizer.probe(
        params={"hidden_layer_sizes_1": 540,
                "hidden_layer_sizes_2": 280},
        lazy=True,
        )

    #load_logs(optimizer, logs=["./Results/nn_f1_macro.json"])

    logger = JSONLogger(path="./Results/nn_f1_macro_try_log.json")

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=0, init_points=0)

    print("Final result:", optimizer.max)


# #############################################################################
# RANDOM FOREST

def rfc_cv(n_estimators, min_samples_split, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """

    # Estimador del Random Forest
    estimator = OneVsRestClassifier(RFC(
        n_estimators=n_estimators,

        min_samples_split=min_samples_split,
        n_jobs=8,
        random_state=42), n_jobs=-1)

    skf = StratifiedKFold(n_splits=-1)

    # Cross validation not neg_log_loss
    cval = cross_val_score(estimator, data, targets,
                           scoring='f1_macro', cv=skf, n_jobs=-1)
    return cval.mean()


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def rfc_crossval(n_estimators, min_samples_split):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """



        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (600, 1300),
            "min_samples_split": (2, 150),
        },
        random_state=41,
        verbose=2
    )

    optimizer.probe(
        params={"n_estimators": 990,
                "min_samples_split": 60},
        lazy=True,
        )

    load_logs(optimizer, logs=["./Results/rf_ovr_f1_final.json"])

    logger = JSONLogger(path="./Results/rf_ovr_f1_final_log.json")

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=0, init_points=0)

    print("Final result:", optimizer.max)


# #############################################################################
# SUPPORT VECTOR MACHINE

def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    # Estimador del SVM
    estimator = OneVsRestClassifier(SVC(kernel='rbf', C=C,
                                        gamma=gamma, probability=False),
                                    n_jobs=-1)

    skf = StratifiedKFold(n_splits=4)

    # Cross validation
    cval = cross_val_score(estimator=estimator, X=data, y=targets, n_jobs=-1,
                           scoring='f1_macro', cv=skf)
    return cval.mean()


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
    
    optimizer.probe(
        params={"expC": 30,
                "expGamma": 2},
        lazy=True,
        )
    
    optimizer.probe(
        params={"expC": 10,
                "expGamma": 0},
        lazy=True,
        )
    
    optimizer.probe(
        params={"expC": 28,
                "expGamma": -10},
        lazy=True,
        )
    #load_logs(optimizer, logs=["./Results/svc_f1_macro.json"])

    logger = JSONLogger(path="./Results/svm_ovr_f1_final_log.json")

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=0, init_points=0)

    print("Final result:", optimizer.max)
