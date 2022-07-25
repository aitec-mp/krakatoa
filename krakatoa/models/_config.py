# -*- coding: utf-8 -*-

'''
Configuration file(:mod:`krakatoa._config`)
============================================================
'''

# ============================================================
# Imports
# ============================================================
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor

import numpy as np

# ============================================================
# Configuration
# ============================================================
def getConfig():
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
                    'vals': [],
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
                    'vals': np.array([0, 0.5, 1]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : float,
                    'min': 0,
                    'max': 1
                }
            },
        },
        'EXTRA_TREE': {
            'type': 'tree',
            'name': 'extraTree',
            'estimator': ExtraTreeRegressor(),
            'attr': {
                'n_estimators': {
                    'vals': [],
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
                    'vals': np.array([0, 1, 10]),
                    'iterable': False,
                    'type': 'num',
                    'ntype' : int,
                    'min': 0,
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
                    'vals': [],
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
                    'min': 0.0,
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
