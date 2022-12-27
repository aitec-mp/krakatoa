# -*- coding: utf-8 -*-

'''
Quick Models(:mod:`krakatoa.models.quick`)
============================================================
'''

#============================================================
# Imports
#============================================================

from sklearn.model_selection import cross_validate

from ._getmodels import getModels
from ._metrics import getScores
import pandas as pd

#============================================================
# Regression Functions
#============================================================

class Regressor():
    def __init__(self):

        self._regression_scores = getScores('regression')

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
    
  