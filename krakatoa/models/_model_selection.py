# -*- coding: utf-8 -*-

'''
Metrics file(:mod:`krakatoa.models._model_selection`)
============================================================
'''

#============================================================
# Imports
#============================================================

from sklearn import model_selection

#============================================================
# Metrics
#============================================================
#TODO: Criar modulo para metrics

def getModel(model):

    models = {
        'train_test_split' : {'f' : model_selection.train_test_split, 'name' : 'train_test_split', 'method' : 0},
        'KFold' : {'f' : model_selection.KFold, 'name' : 'KFold', 'method' : 1},
        # 'GroupKFold' : {'f' : model_selection.GroupKFold, 'name' : 'GroupKFold', 'method' : 1},
        'RepeatedKFold' : {'f' : model_selection.RepeatedKFold, 'name' : 'RepeatedKFold', 'method' : 1},
        'StratifiedGroupKFold' : {'f' : model_selection.StratifiedGroupKFold, 'name' : 'StratifiedGroupKFold', 'method' : 1},
        'cross_validate' : {'f' : model_selection.cross_validate, 'name' : 'cross_validate', 'method' : 2},
        'cross_val_score' : {'f' : model_selection.cross_val_score, 'name' : 'cross_val_score', 'method' : 2}
        }
    
    return models.get(model, None)