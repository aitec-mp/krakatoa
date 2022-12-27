# -*- coding: utf-8 -*-

'''
Metrics file(:mod:`krakatoa.models._metrics`)
============================================================
'''

#============================================================
# Imports
#============================================================

from sklearn import metrics

#============================================================
# Metrics
#============================================================
#TODO: Criar modulo para metrics

def getScores(kind):

    regression_scores = {
        'explained_variance' : {'f' : metrics.explained_variance_score, 'name' : 'explained_variance'},
        'max_error' : {'f' : metrics.max_error, 'name' : 'max_error'},
        'mae' : {'f' : metrics.mean_absolute_error, 'name' : 'neg_mean_absolute_error'},
        'mse' : {'f' : metrics.mean_squared_error, 'name' : 'neg_mean_squared_error'},
        'rmse' : {'f' : metrics.mean_squared_error, 'name' : 'neg_root_mean_squared_error'}, #TODO: criar funcao RMSE,
        'msle' : {'f' : metrics.mean_squared_log_error, 'name' : 'neg_mean_squared_log_error'},
        'neg_median_absolute_error' : {'f' : metrics.median_absolute_error, 'name' : 'neg_median_absolute_error'},
        'r2' : {'f' : metrics.r2_score, 'name' : 'r2'},
        'poison' : {'f' : metrics.mean_poisson_deviance, 'name' : 'neg_mean_poisson_deviance'},
        'gama' : {'f' : metrics.mean_gamma_deviance, 'name' : 'neg_mean_gamma_deviance'},
        'neg_mean_absolute_percentage_error' : {'f' : metrics.mean_absolute_percentage_error, 'name' : 'neg_mean_absolute_percentage_error'},
        'd2_absolute_error_score' : {'f' : metrics.d2_absolute_error_score, 'name' : 'd2_absolute_error_score'},
        'd2_pinball_score' : {'f' : metrics.d2_pinball_score, 'name' : 'd2_pinball_score'},
        'd2_tweedie_score' : {'f' : metrics.d2_tweedie_score, 'name' : 'd2_tweedie_score'}
        }


    classification_scores = {
        'accuracy' : {'f' : metrics.accuracy_score, 'name' : 'accuracy'},
        'balanced_accuracy' : {'f' : metrics.balanced_accuracy_score, 'name' : 'balanced_accuracy'},
        'roc_auc' : {'f' : metrics.roc_auc_score, 'name' : 'roc_auc'}
        }

    if kind == "classification":
        scores = classification_scores
    else:
        scores = regression_scores
    
    return scores
