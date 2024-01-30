# -*- coding: utf-8 -*-

'''
Quick Models(:mod:`krakatoa.models.quick`)
============================================================
'''

#============================================================
# Imports
#============================================================

from sklearn.model_selection import cross_validate, train_test_split, cross_val_score


from ._getmodels import getModels
from ._metrics import getScores
from ._model_selection import getModel
import pandas as pd

import time
#============================================================
# Regression Functions
#============================================================

class Regressor():
    def __init__(self):

        self._regression_scores = getScores('regression')


    def _runModels(self, models, x, y, scoring, cv, model_selection='cross_validate', metrics = ['r2'], model_selection_params = {}):
        results = {'estimator' : [], 'fit_time' : [], 'score_time' : []}

        # Model selection part
        res_model_selection = getModel(model=model_selection)

        # model_selection_function
        modelSelFunc = res_model_selection['f']
        method = res_model_selection['method']
        if method == 0:
            # For methods that returns x_train, x_test, y_train, y_test
            
            X_train, X_test, y_train, y_test = modelSelFunc(x, y, **model_selection_params)
            dict_metrics = getScores('regression')
            for model in models:
                results['estimator'].append(model['name'])
                # Fit time
                fit_start_time = time.perf_counter()
                estimator = model['estimator']
                estimator.fit(X_train, y_train)
                fit_end_time = time.perf_counter()

                results['fit_time'].append(fit_end_time - fit_start_time)

                y_predicted = estimator.predict(X_test)

                metric_start_time = time.perf_counter()
                for metric in metrics:
                    dict_metric = dict_metrics.get(metric, None)
                    if dict_metric == None:
                        break

                    res_metric = dict_metric['f'](y_test, y_predicted)
                    metric_name = f'{metric}_score'
                    if metric_name not in results.keys():
                        results[metric_name] = []
                    results[metric_name].append(res_metric)

                metric_end_time = time.perf_counter()
                metric_time = metric_end_time - metric_start_time
                results['score_time'] = metric_time

        elif method == 1:
            # Must improve implementation in order to use other KFolds methods such as Group and Stratified
            folds = modelSelFunc(**model_selection_params)
            for model in models:

                res = cross_validate(model['estimator'], x, y, scoring=scoring, cv=folds, n_jobs=-1)
                results['estimator'].append(model['name'])

                for k, v in res.items():
                    if k not in results.keys():
                        results[k] = []
                        
                    results[k].append(v.mean())

        elif method == 2:

            for model in models:

                res = modelSelFunc(model['estimator'], x, y, scoring=scoring, cv=cv, n_jobs=-1)
                results['estimator'].append(model['name'])

                for k, v in res.items():
                    if k not in results.keys():
                        results[k] = []
                        
                    results[k].append(v.mean())
        
        return results

    def _runModelsYield(self, models, x, y, scoring, cv, model_selection='cross_validate', metrics = ['r2'], model_selection_params = {}):
        
        results = {'estimator' : [], 'fit_time' : [], 'score_time' : []}

        # Model selection part
        res_model_selection = getModel(model=model_selection)

        # model_selection_function
        modelSelFunc = res_model_selection['f']
        method = res_model_selection['method']
        if method == 0:
            # For methods that returns x_train, x_test, y_train, y_test
            
            X_train, X_test, y_train, y_test = modelSelFunc(x, y, **model_selection_params)
            dict_metrics = getScores('regression')
            for model in models:
                yield {
                        'status_code' : 1,
                        'status' : 'running',
                        'message' : 'Training running',
                        'data' : model['name']
                        }
                
                results['estimator'].append(model['name'])
                # Fit time
                fit_start_time = time.perf_counter()
                estimator = model['estimator']
                estimator.fit(X_train, y_train)
                fit_end_time = time.perf_counter()

                results['fit_time'].append(fit_end_time - fit_start_time)

                y_predicted = estimator.predict(X_test)

                metric_start_time = time.perf_counter()
                for metric in metrics:
                    dict_metric = dict_metrics.get(metric, None)
                    if dict_metric == None:
                        break

                    res_metric = dict_metric['f'](y_test, y_predicted)
                    metric_name = f'{metric}_score'
                    if metric_name not in results.keys():
                        results[metric_name] = []
                    results[metric_name].append(res_metric)

                metric_end_time = time.perf_counter()
                metric_time = metric_end_time - metric_start_time
                results['score_time'] = metric_time

        elif method == 1:
            # Must improve implementation in order to use other KFolds methods such as Group and Stratified
            folds = modelSelFunc(**model_selection_params)
            for model in models:
                yield {
                        'status_code' : 1,
                        'status' : 'running',
                        'message' : 'Training running',
                        'data' : model['name']
                        }
                
                res = cross_validate(model['estimator'], x, y, scoring=scoring, cv=folds, n_jobs=-1)
                results['estimator'].append(model['name'])

                for k, v in res.items():
                    if k not in results.keys():
                        results[k] = []
                        
                    results[k].append(v.mean())

        elif method == 2:

            for model in models:
                yield {
                        'status_code' : 1,
                        'status' : 'running',
                        'message' : 'Training running',
                        'data' : model['name']
                        }
                
                res = modelSelFunc(model['estimator'], x, y, scoring=scoring, cv=cv, n_jobs=-1)
                results['estimator'].append(model['name'])

                for k, v in res.items():
                    if k not in results.keys():
                        results[k] = []
                        
                    results[k].append(v.mean())
        
        yield {
            'status_code' : 0,
            'status' : 'finished',
            'message' : 'Training finished',
            'data' : results
        }

    def linearRegression(self, x, y, score=['r2'], cv=5, **kwargs):
        '''
        Quick linear regression models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['r2'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''
        
        
        models = getModels(mode='regression', modelClasses=['linear'])
        
        scoring = []
        for s in score:
            scoring.append(self._regression_scores[s]['name'])
            # results['test_' + regression_scores[s]['name']]= []
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)

    def treeRegression(self, x, y, score=['r2'], cv=5, **kwargs):
        '''
        Quick linear regression models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['r2'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''
        
        models = getModels(mode='regression', modelClasses=['tree'])
        
        scoring = []
        for s in score:
            scoring.append(self._regression_scores[s]['name'])
            # results['test_' + regression_scores[s]['name']]= []
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)

    def boostRegression(self, x, y, score=['r2'], cv=5, **kwargs):
        '''
        Quick linear regression models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['r2'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''

        models = getModels(mode='regression', modelClasses=['boost'])
        
        scoring = []
        for s in score:
            scoring.append(self._regression_scores[s]['name'])
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)

    def multiRegression(self, x, y, models = ['boost', 'linear', 'tree'], score=['r2'], cv=5):
        
        models = getModels(mode='regression', modelClasses=models)
        
        scoring = []
        for s in score:
            scoring.append(self._regression_scores[s]['name'])
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)
    
    def customRegression(self, x, y, models = ['boost', 'linear', 'tree'], selMode='type', score=['r2'], cv=5, model_selection='cross_validate', model_selection_params = {}, verbose_yield:bool = False):
        
        models = getModels(mode='regression', modelClasses=models, selMode=selMode)

        scoring = []
        for s in score:
            scoring.append(self._regression_scores[s]['name'])
            
        if verbose_yield:
            for i in self._runModelsYield(models, x, y, scoring, cv, model_selection=model_selection, metrics=score, model_selection_params=model_selection_params):
                yield i
        else:
            results = self._runModels(models, x, y, scoring, cv, model_selection=model_selection, metrics=score, model_selection_params=model_selection_params)
        
            return pd.DataFrame(results)
    

#============================================================
# Classification Functions
#============================================================


class Classifier():

    def __init__(self):

        self._classification_scores = getScores('classification')

    def _crossvalidate(self, estimator, x, y, scoring, cv=5):

        result = cross_validate(estimator, x, y, scoring=scoring, cv=cv, n_jobs=-1)
        
        return result

    def _runModels(self, models, x, y, scoring, cv):
        results = {'estimator' : [], 'fit_time' : [], 'score_time' : []}
        
        for model in models:
            res = self._crossvalidate(model['estimator'], x, y, scoring, cv)
            
            results['estimator'].append(model['name'])

            for k, v in res.items():
                if k not in results.keys():
                    results[k] = []
                    
                results[k].append(v.mean())
        
        return results

    def svmClassifier(self, x, y, score=['accuracy'], cv=5, **kwargs):
        '''
        Quick SVM classification models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['accuracy'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''
        
        models = getModels(mode='classification', modelClasses=['svm'])
        
        scoring = []
        for s in score:
            scoring.append(self._classification_scores[s]['name'])
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)

    def treeClassifier(self, x, y, score=['accuracy'], cv=5, **kwargs):
        '''
        Quick Tree classification models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['accuracy'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''
        
        models = getModels(mode='classification', modelClasses=['tree'])
        
        scoring = []
        for s in score:
            scoring.append(self._classification_scores[s]['name'])
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)

    def boostClassifier(self, x, y, score=['accuracy'], cv=5, **kwargs):
        '''
        Quick Boost classification models evaluation.

        Parameters
        ----------
        x : DATAFRAME
            Dataset Features.
        y : NUMPY ARRAY
            Target.
        score : LIST, optional
            Scoring metrics. The default is ['accuracy'].
        cv : INT, optional
            Number of kfolds. The default is 5.

        Returns
        -------
        return : DATAFRAME
            Returns dataframe with models and selected metrics score.

        '''
        
        models = getModels(mode='classification', modelClasses=['boost'])
        
        scoring = []
        for s in score:
            scoring.append(self._classification_scores[s]['name'])
            
        results = self._runModels(models, x, y, scoring, cv)
        
        return pd.DataFrame(results)
    
  