# -*- coding: utf-8 -*-

'''
Configuration file(:mod:`krakatoa._config`)
============================================================
'''

# ============================================================
# Imports
# ============================================================
from pickle import NONE
from telnetlib import KERMIT
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import numpy as np

# ============================================================
# Configuration
# ============================================================
def getConfigRegr():
    
    conf = {
        'LINEAR': {
            'type': 'linear',
            'name': 'linear',
            'estimator': LinearRegression(),
            'attr': {}
        },
        'RIDGE': {
            'type': 'linear',
            'name': 'ridge',
            'estimator': Ridge(),
            'attr': {
                'alpha': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                }
            },
        },
        'LASSO': {
            'type': 'linear',
            'name': 'lasso',
            'estimator': Lasso(),
            'attr': {
                'alpha': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'tol': {
                    'vals': np.array([0.001, 0.01, 0.1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                }

            },
        },
        'ELASTIC_NET': {
            'type': 'linear',
            'name': 'elasticnet',
            'estimator': ElasticNet(),
            'attr': {
                'alpha': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'l1_ratio': {
                    'vals': np.array([0.1, 0.5, 0.9]),
                    'iterable': False,
                    'ntype' : float,
                    'type': 'num',
                    'min': 0,
                    'max': 1
                },
                'tol': {
                    'vals': np.array([0.001, 0.01, 0.1]),
                    'iterable': False,
                    'ntype' : float,
                    'type': 'num',
                    'min': 0,
                    'max': None
                }
            },
        },
        'DECISION_TREE': {
            'type': 'tree',
            'name': 'decisionTree',
            'estimator': DecisionTreeRegressor(),
            'attr': {
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': 1
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 5, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 1,
                    'max': None
                }
            },
        },
        'RANDOM_FOREST': {
            'type': 'tree',
            'name': 'randomForest',
            'estimator': RandomForestRegressor(),
            'attr': {
                'n_estimators': {
                    'vals': [100],
                    'iterable': True,
                    'init_val': 100,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 5,
                    'max': None
                },
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 3, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                }
            },
        },
        'EXTRA_TREE': {
            'type': 'tree',
            'name': 'extraTree',
            'estimator': ExtraTreeRegressor(),
            'attr': {
                # 'n_estimators': {
                #     'vals': [100],
                #     'iterable': True,
                #     'init_val': 100,
                #     'def_step': 100,
                #     'type': 'num',
                #     'ntype' : int,
                #     'min': 5,
                #     'max': None
                # },
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 5, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 1,
                    'max': None
                }
            },
        },
        'ADABOOST': {
            'type': 'boost',
            'name': 'adaboost',
            'estimator': AdaBoostRegressor(),
            'attr': {
                'n_estimators': {
                    'vals': [50],
                    'iterable': True,
                    'init_val': 50,
                    'def_step': 50,
                    'type': 'num',
                    'ntype' : int,
                    'min': 5,
                    'max': None
                },
                'learning_rate': {
                    'vals': np.array([0.001, 0.1, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': None
                },
                'loss': {
                    'vals': ['linear', 'square', 'exponential'],
                    'iterable': False,
                    'type': 'cat',
                    'min': 0.0,
                    'max': None
                }
            }
        },
        'XGBOOST': {
            'type': 'boost',
            'name': 'xgboost',
            'estimator': XGBRegressor(),
            'attr': {
                'eta': {
                    'vals': np.array([0.1, 0.3, 0.5]),
                    'default': 0.3,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': 1.0
                },
                'gamma': {
                    'vals': np.array([0, 1, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([2, 6, 15]),
                    'default': 6,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'min_child_weight': {
                    'vals': np.array([0, 5, 10]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0.0,
                    'max': None
                },
                'max_delta_step': {
                    'vals': np.array([0, 5, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,                   
                    'min': 0.0,
                    'max': None
                },
                'subsample': {
                    'vals': np.array([0, 0.5, 1]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0.0,
                    'max': 1.0
                },
                'lambda': {  # L2 regularizarion
                    'vals': np.array([1, 10, 100]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'alpha': {  # L1 regularizarion
                    'vals': np.array([0, 10, 100]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'max_leaves': {
                    'vals': np.array([0, 5, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'loss': {
                    'vals': ['linear', 'square', 'exponential'],
                    'iterable': False,
                    'type': 'cat',
                    'min': 0.0,
                    'max': None
                }
            },
        }
    }

    return conf

def getConfigClass():
    
    conf = {
        'SVC_LINEAR': {
            'type': 'svm',
            'name': 'svc_linear',
            'estimator': SVC(kernel='linear'),
            'attr': {
                'C': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'tol': {
                    'vals': np.array([0.00001, 0.0001, 0.01]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.000000001,
                    'max': None
                },
                'max_iter': {
                    'vals': [500],
                    'iterable': True,
                    'init_val': 500,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 100,
                    'max': None
                }
            }
        },
        'SVC_RBF': {
            'type': 'svm',
            'name': 'svc_rbf',
            'estimator': SVC(kernel='rbf'),
            'attr': {
                'C': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'tol': {
                    'vals': np.array([0.00001, 0.0001, 0.01]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.000000001,
                    'max': None
                },
                'max_iter': {
                    'vals': [500],
                    'iterable': True,
                    'init_val': 500,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 100,
                    'max': None
                }
            }
        },
        'SVC_POLY': {
            'type': 'svm',
            'name': 'svc_poly',
            'estimator': SVC(kernel='poly'),
            'attr': {
                'C': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'degree' : {
                    'vals': [2],
                    'iterable': True,
                    'init_val': 2,
                    'def_step': 1,
                    'type': 'num',
                    'ntype' : int,
                    'min': 2,
                    'max': 5
                },
                'coef0': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': None
                },
                'tol': {
                    'vals': np.array([0.00001, 0.0001, 0.01]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.000000001,
                    'max': None
                },
                'max_iter': {
                    'vals': [500],
                    'iterable': True,
                    'init_val': 500,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 100,
                    'max': None
                }
            }
        },
        'SVC_SIGMOID': {
            'type': 'svm',
            'name': 'svc_sigmoid',
            'estimator': SVC(kernel='sigmoid'),
            'attr': {
                'C': {
                    'vals': np.array([0.1, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'degree' : {
                    'vals': [2],
                    'iterable': True,
                    'init_val': 2,
                    'def_step': 1,
                    'type': 'num',
                    'ntype' : int,
                    'min': 2,
                    'max': 5
                },
                'coef0': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': None
                },
                'tol': {
                    'vals': np.array([0.00001, 0.0001, 0.01]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.000000001,
                    'max': None
                },
                'max_iter': {
                    'vals': [500],
                    'iterable': True,
                    'init_val': 500,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 100,
                    'max': None
                }
            }
        },
        'RANDOM_FOREST': {
            'type': 'tree',
            'name': 'randomForest',
            'estimator': RandomForestClassifier(),
            'attr': {
                'n_estimators': {
                    'vals': [100],
                    'iterable': True,
                    'init_val': 100,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 5,
                    'max': None
                },
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 3, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                }
            },
        },
        'EXTRA_TREE': {
            'type': 'tree',
            'name': 'extraTree',
            'estimator': ExtraTreesClassifier(),
            'attr': {
                'n_estimators': {
                    'vals': [100],
                    'iterable': True,
                    'init_val': 100,
                    'def_step': 100,
                    'type': 'num',
                    'ntype' : int,
                    'min': 5,
                    'max': None
                },
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 5, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 1,
                    'max': None
                }
            },
        },
        'DECISION_TREE': {
            'type': 'tree',
            'name': 'decisionTree',
            'estimator': DecisionTreeClassifier(),
            'attr': {
                'min_samples_split': {
                    'vals': np.array([0.1, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': 1
                },
                'min_samples_leaf': {
                    'vals': np.array([0.1, 1, 5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.1,
                    'max': None
                },
                'min_weight_fraction_leaf': {
                    'vals': np.array([0.0, 0.25, 0.5]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'min_impurity_decrease': {
                    'vals': np.array([0.0, 0.5, 1.0]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([1, 5, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 1,
                    'max': None
                }
            },
        },
        'ADABOOST': {
            'type': 'boost',
            'name': 'adaboost',
            'estimator': AdaBoostClassifier(),
            'attr': {
                'n_estimators': {
                    'vals': [50],
                    'iterable': True,
                    'init_val': 50,
                    'def_step': 50,
                    'type': 'num',
                    'ntype' : int,
                    'min': 5,
                    'max': None
                },
                'learning_rate': {
                    'vals': np.array([0.001, 0.1, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': None
                },
                'loss': {
                    'vals': ['linear', 'square', 'exponential'],
                    'iterable': False,
                    'type': 'cat',
                    'min': 0.0,
                    'max': None
                }
            }
        },
        'XGBOOST': {
            'type': 'boost',
            'name': 'xgboost',
            'estimator': XGBClassifier(),
            'attr': {
                'eta': {
                    'vals': np.array([0.1, 0.3, 0.5]),
                    'default': 0.3,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0.0,
                    'max': 1.0
                },
                'gamma': {
                    'vals': np.array([0, 1, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'max_depth': {
                    'vals': np.array([2, 6, 15]),
                    'default': 6,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'min_child_weight': {
                    'vals': np.array([0, 5, 10]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0.0,
                    'max': None
                },
                'max_delta_step': {
                    'vals': np.array([0, 5, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,                   
                    'min': 0.0,
                    'max': None
                },
                'subsample': {
                    'vals': np.array([0, 0.5, 1]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0.0,
                    'max': 1.0
                },
                'lambda': {  # L2 regularizarion
                    'vals': np.array([1, 10, 100]),
                    'default': 1,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'alpha': {  # L1 regularizarion
                    'vals': np.array([0, 10, 100]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'max_leaves': {
                    'vals': np.array([0, 5, 10]),
                    'default': 0,
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
                    'max': None
                },
                'loss': {
                    'vals': ['linear', 'square', 'exponential'],
                    'iterable': False,
                    'type': 'cat',
                    'min': 0.0,
                    'max': None
                }
            },
        }



    }

    return conf

