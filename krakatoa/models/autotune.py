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
        self._regression_scores = getScores('regression')

    def fit(self, estimator, x, y, score=['r2'], cv=3, verbose=0, lr=0.001, random_state=0):
        results = {}
        modelConfig = getModels(mode='regression', modelClasses=[estimator], selMode='name', random_state=random_state)

        #Collector variables
        models = np.array([])
        opt_params = dict()
        scores = np.array([])
        
        if len(modelConfig) > 0:
            #Model definition
            # model = model[0]['estimator']
            modelConfig = modelConfig[0]
            model = modelConfig.get('estimator', '')

            scoring = []
            for s in score:
                scoring.append(self._regression_scores[s]['name'])
                            
            #carrega parametros
            # num_parameters = {key:attr['vals'] for key, attr in modelConfig['attr'].items() if attr['type'] == 'num' and attr['iterable'] == False}
            # num_parameters = {key:attr for key, attr in modelConfig['attr'].items() if attr['type'] == 'num' and attr['iterable'] == False}
            daux = {key:attr for key, attr in modelConfig['attr'].items() if attr['type'] == 'num'}

            # Alterado para que seja uma lista ordenada | pois dict nao pode ser |
            num_parameters = []
            for k, p in daux.items():
                p['key'] = k
                num_parameters.append(p)


            num_parameters = sorted(num_parameters, key= lambda item:item['iterable'])

            best_score = 0
            best_params = {}
            # for k, p in num_parameters.items(): 
            for item in num_parameters: 
 
                #Specific attr cfg
                # paramCfg = modelConfig['attr'][k]
                paramCfg = modelConfig['attr'][item['key']]

                stop = False
                #Process once to discover the best of initial three parameters
                grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:item['vals']}
                grid = GridSearchCV(model, param_grid=grid_params, cv=cv, n_jobs=-1, verbose=0)
                grid.fit(x, y)
                
                # Verifica se aplicando o grid search melhora resultados
                # se nao melhorar nao entra no pipeline para descobrir melhores parametros
                if grid.best_score_ > best_score:
                    
                    best_score = grid.best_score_
                    best_params = grid.best_params_
                    
                    last_best_model = grid.best_estimator_
                    last_best_score = best_score
                    last_best_param = best_params
                    
                else:
                    stop = True
                    
                if item['iterable'] == False:
                    param_range = item['vals']
                    
                    if verbose > 0:
                        print(f'-------- {item["key"]} ----------')
                    # TODO> Abstrair para um função             
                    while stop == False:
                        # if last_best_param[k] == None:
                        if last_best_param[item['key']] == None:
                            stop = True
                            break

                        # spec_best_param = last_best_param[k]
                        spec_best_param = last_best_param[item['key']]
                        
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

                        # Calculo dos novos limites a esquerda e direita    
                        aux_param_l = (spec_best_param - lim_l)/2
                        aux_param_r = (lim_r - spec_best_param)/2 + spec_best_param

                        param_l = aux_param_l if aux_param_l >= 0 else lim_l
                        param_r = aux_param_r if aux_param_r >= 0 else lim_r

                        # Verifica o tipo da variavel, se for INT devemos garantir que nao esteja passando como float e resolver

                        # if isinstance(paramCfg['ntype'], int):
                        if paramCfg['ntype'] == int:
                            param_l = int(math.ceil(param_l))
                            param_r = int(math.ceil(param_r))


                        #re do parameter range
                        param_range = np.array([param_l, spec_best_param , param_r])
                        
                        _grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:param_range}
                        _grid = GridSearchCV(model, param_grid=_grid_params, cv=cv, n_jobs=-1, verbose=0)
                        _grid.fit(x, y)
                        
                        _best_score = _grid.best_score_
                        _best_params = _grid.best_params_
                        _best_model = _grid.best_estimator_
                        
                        #  Show verbose if it is necessary
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
                            # last_best_param[k] = _best_params[k]
                            last_best_param[item['key']] = _best_params[item['key']]
                        
                            if _diff <= lr:
                                stop = True
                                
                        else:
                            stop = True

                # Trecho para iteraveis
                # TODO melhorar essa parte deixando mais robusto
                else:
                    if verbose > 0:
                        print(f'-------- {item["key"]} ----------')
                    min_try = 5 # Quantidade minima de tentativas  antes de sair do loop
                    value = item['init_val']
                    step = item['def_step']
                    maxVal = item['max']
                    hist_value = [] #tuple com value e accuracy

                    stop = False
                    try_count = 1

                    while stop == False:

                        _grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:[value]}
                        _grid = GridSearchCV(model, param_grid=_grid_params, cv=cv, n_jobs=-1, verbose=0)
                        _grid.fit(x, y)
                        
                        _best_score = _grid.best_score_
                        _best_params = _grid.best_params_
                        _best_model = _grid.best_estimator_
                        
                        print("_best_score")
                        print(_best_score)
                        hist_value.append((value, _best_score))
                        #  Show verbose if it is necessary
                        if verbose > 0:
                            
                            print("-----------------")
                            print("Value =====> ", value)
                            print("opt params: ", _grid_params)
                            print("Score: ", _best_score)
                            print("Params: ", _best_params)
                            print("Diff: ", (_best_score - last_best_score))
                            print("-----------------")
                            
                            
                        _diff = _best_score - last_best_score
                        
                        if last_best_score < _best_score:
                            last_best_model = _best_model
                            last_best_score = _best_score
                            # last_best_param[k] = _best_params[k]
                            last_best_param[item['key']] = _best_params[item['key']]
                    
                        if (_diff <= lr and try_count >= min_try) or (maxVal == value):
                            stop = True
                            print(hist_value)
                            # maxValue = max(hist_value, key=lambda item:item[1])
                        elif _diff >= lr and (min_try - try_count) < 2:
                            print("reset")
                            try_count = 0
                            min_try = 3 #Zera o contador e deixa pelo menor 3 a mais para tentar

                        try_count += 1
                        value += step

                
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
    
    
class ClassifierAF():

    def __init__(self):
        self._classification_score = getScores('classification')

    def fit(self, estimator, x, y, score=['accuracy'], cv=3, verbose=0, lr=0.001, random_state=0):
        results = {}
        modelConfig = getModels(mode='classification', modelClasses=[estimator], selMode='name', random_state=random_state)

        #Collector variables
        models = np.array([])
        opt_params = dict()
        scores = np.array([])
        
        if len(modelConfig) > 0:
            #Model definition
            # model = model[0]['estimator']
            modelConfig = modelConfig[0]
            model = modelConfig.get('estimator', '')

            scoring = []
            for s in score:
                scoring.append(self._classification_score[s]['name'])
                            
            #carrega parametros
            # num_parameters = {key:attr['vals'] for key, attr in modelConfig['attr'].items() if attr['type'] == 'num' and attr['iterable'] == False}
            # num_parameters = {key:attr for key, attr in modelConfig['attr'].items() if attr['type'] == 'num' and attr['iterable'] == False}
            daux = {key:attr for key, attr in modelConfig['attr'].items() if attr['type'] == 'num'}

            # Alterado para que seja uma lista ordenada | pois dict nao pode ser |
            num_parameters = []
            for k, p in daux.items():
                p['key'] = k
                num_parameters.append(p)


            num_parameters = sorted(num_parameters, key= lambda item:item['iterable'])

            best_score = 0
            best_params = {}
            # for k, p in num_parameters.items(): 
            for item in num_parameters: 
 
                #Specific attr cfg
                # paramCfg = modelConfig['attr'][k]
                paramCfg = modelConfig['attr'][item['key']]

                stop = False
                #Process once to discover the best of initial three parameters
                grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:item['vals']}
                grid = GridSearchCV(model, param_grid=grid_params, cv=cv, n_jobs=-1, verbose=0)
                grid.fit(x, y)
                
                # Verifica se aplicando o grid search melhora resultados
                # se nao melhorar nao entra no pipeline para descobrir melhores parametros
                if grid.best_score_ > best_score:
                    
                    best_score = grid.best_score_
                    best_params = grid.best_params_
                    
                    last_best_model = grid.best_estimator_
                    last_best_score = best_score
                    last_best_param = best_params
                    
                else:
                    stop = True
                    
                if item['iterable'] == False:
                    param_range = item['vals']
                    
                    if verbose > 0:
                        print(f'-------- {item["key"]} ----------')
                        
                    # TODO> Abstrair para um função             
                    while stop == False:
                        # if last_best_param[k] == None:
                        if last_best_param[item['key']] == None:
                            stop = True
                            break

                        # spec_best_param = last_best_param[k]
                        spec_best_param = last_best_param[item['key']]
                        
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

                        # Calculo dos novos limites a esquerda e direita    
                        aux_param_l = (spec_best_param - lim_l)/2
                        aux_param_r = (lim_r - spec_best_param)/2 + spec_best_param

                        param_l = aux_param_l if aux_param_l >= 0 else lim_l
                        param_r = aux_param_r if aux_param_r >= 0 else lim_r

                        # Verifica o tipo da variavel, se for INT devemos garantir que nao esteja passando como float e resolver

                        # if isinstance(paramCfg['ntype'], int):
                        if paramCfg['ntype'] == int:
                            param_l = int(math.ceil(param_l))
                            param_r = int(math.ceil(param_r))


                        #re do parameter range
                        param_range = np.array([param_l, spec_best_param , param_r])
                        
                        _grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:param_range}
                        _grid = GridSearchCV(model, param_grid=_grid_params, cv=cv, n_jobs=-1, verbose=0)
                        _grid.fit(x, y)
                        
                        _best_score = _grid.best_score_
                        _best_params = _grid.best_params_
                        _best_model = _grid.best_estimator_
                        
                        #  Show verbose if it is necessary
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
                            # last_best_param[k] = _best_params[k]
                            last_best_param[item['key']] = _best_params[item['key']]
                        
                            if _diff <= lr:
                                stop = True
                                
                        else:
                            stop = True

                # Trecho para iteraveis
                # TODO melhorar essa parte deixando mais robusto
                else:
                    if verbose > 0:
                        print(f'-------- {item["key"]} ----------')

                    min_try = 5 # Quantidade minima de tentativas  antes de sair do loop
                    value = item['init_val']
                    step = item['def_step']
                    maxVal = item['max']
                    hist_value = [] #tuple com value e accuracy

                    stop = False
                    try_count = 1

                    while stop == False:

                        _grid_params =  {**{_k:np.array([_v]) for _k, _v in opt_params.items()}, item['key']:[value]}
                        _grid = GridSearchCV(model, param_grid=_grid_params, cv=cv, n_jobs=-1, verbose=0)
                        _grid.fit(x, y)
                        
                        _best_score = _grid.best_score_
                        _best_params = _grid.best_params_
                        _best_model = _grid.best_estimator_
                        
                        print("_best_score")
                        print(_best_score)
                        hist_value.append((value, _best_score))
                        #  Show verbose if it is necessary
                        if verbose > 0:
                            
                            print("-----------------")
                            print("Value =====> ", value)
                            print("opt params: ", _grid_params)
                            print("Score: ", _best_score)
                            print("Params: ", _best_params)
                            print("Diff: ", (_best_score - last_best_score))
                            print("-----------------")
                            
                            
                        _diff = _best_score - last_best_score
                        
                        if last_best_score < _best_score:
                            last_best_model = _best_model
                            last_best_score = _best_score
                            # last_best_param[k] = _best_params[k]
                            last_best_param[item['key']] = _best_params[item['key']]
                    
                        if (_diff <= lr and try_count >= min_try) or (maxVal == value):
                            stop = True
                            print(hist_value)
                            # maxValue = max(hist_value, key=lambda item:item[1])
                        elif _diff >= lr and (min_try - try_count) < 2:
                            print("reset")
                            try_count = 0
                            min_try = 3 #Zera o contador e deixa pelo menor 3 a mais para tentar

                        try_count += 1
                        value += step

                
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
