# -*- coding: utf-8 -*-

'''
AutoTune models(:mod:`krakatoa.models.autotune`)
============================================================
'''

#============================================================
# Imports
#============================================================

from sklearn.model_selection import GridSearchCV
from ._getmodels import getModels
from ._metrics import getScores

import numpy as np
import math

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

#============================================================
# Regression Functions
#============================================================

@ignore_warnings(category=ConvergenceWarning)
class RegressorAF():

    def __init__(self):
        self._regression_scores = getScores()

    def fit(self, estimator, x, y, score=['r2'], cv=3, verbose=0, lr=0.001, random_state=0):
        results = {}
        modelConfig = getModels([estimator], 'name', random_state)
            
        if len(modelConfig) > 0:
            #Model definition
            # model = model[0]['estimator']
            modelConfig = modelConfig[0]
            model = modelConfig.get('estimator', '')

            scoring = []
            for s in score:
                scoring.append(self._regression_scores[s]['name'])
                
            #Collector variables
            models = np.array([])
            opt_params = dict()
            scores = np.array([])
            
            #carrega parametros
            num_parameters = {key:attr['vals'] for key, attr in modelConfig['attr'].items() if attr['type'] == 'num' and attr['iterable'] == False}
            
            best_score = 0
            best_params = {}
            for k, p in num_parameters.items(): 
                #Specific attr cfg
                paramCfg = modelConfig['attr'][k]
                
                stop = False
                #Process once to discover the best of initial three parameters
                grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, k:p}
                
                grid = GridSearchCV(model, param_grid=grid_params, cv=cv, n_jobs=-1, verbose=0)
                grid.fit(x, y)
                
                if grid.best_score_ > best_score:
                    
                    best_score = grid.best_score_
                    best_params = grid.best_params_
                    
                    last_best_model = grid.best_estimator_
                    last_best_score = best_score
                    last_best_param = best_params
                    
                else:
                    stop = True
                    
                param_range = num_parameters[k]
                
                if verbose > 0:
                    print(f'-------- {k} ----------')
                    
                                
                while stop == False:
                    if last_best_param[k] == None:
                        stop = True
                        break

                    spec_best_param = last_best_param[k]
                    
                    #identify if the value is in the center, left or right
                    index = np.where(param_range == spec_best_param)[0][0]
                    #left
                    if index == 0: 
                        lim_r = param_range[1]
                        lim_l = 0
                    elif index == 1:
                        lim_r = param_range[2]
                        lim_l = param_range[0]
                    else:
                        lim_r = spec_best_param*2 #TODO: Melhorar logica 
                        lim_l = param_range[1]
                                                        
                    aux_param_l = (spec_best_param - lim_l)/2
                    aux_param_r = (lim_r - spec_best_param)/2 + spec_best_param
                    
                    param_l = aux_param_l if aux_param_l >= 0 else lim_l
                    param_r = aux_param_r if aux_param_r >= 0 else lim_r

                    if isinstance(paramCfg['ntype'], int):
                        param_l = math.ceil(param_l)
                        param_r = math.ceil(param_r)

                    #re do parameter range
                    param_range = np.array([param_l, spec_best_param , param_r])
                    
                    _grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, k:param_range}
                    _grid = GridSearchCV(model, param_grid=_grid_params, cv=cv, n_jobs=-1, verbose=0)
                    _grid.fit(x, y)
                    
                    _best_score = _grid.best_score_
                    _best_params = _grid.best_params_
                    _best_model = _grid.best_estimator_
                    
                    if verbose > 0:
                        
                        print("-----------------")
                        print("Range =====> ", param_range)
                        print("opt params: ", _grid_params)
                        print("Score: ", _best_score)
                        print("Params: ", _best_params)
                        print("Diff: ", (_best_score - last_best_score))
                        print("-----------------")
                        
                    if _best_score > last_best_score:
                        _diff = _best_score - last_best_score
                        
                        last_best_model = _best_model
                        last_best_score = _best_score
                        last_best_param[k] = _best_params[k]
                    
                        if _diff <= lr:
                            stop = True
                            
                    else:
                        stop = True

                #Store best params and results

                models = np.append(models, last_best_model)
                opt_params = {**opt_params, **last_best_param}
                scores = np.append(scores, last_best_score)
                    
        
        results = {
            'models' : models,
            'best_params' : opt_params,
            'scores' : scores
            }

        return results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
