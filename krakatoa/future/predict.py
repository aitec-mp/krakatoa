# -*- coding: utf-8 -*-

'''
Prediction module (:mod:`krakatoa.future.predict`)
============================================================
'''

# ============================================================
# Imports
# ============================================================
import pickle
import joblib
import numpy as np
# ============================================================


class Predictor():

    def __init__(self):
        self.model = None

    def loadModel(self, model=None, model_path = None, loader_engine: str = 'pickle'):

        if model is None:

            if loader_engine == 'pickle':
                with open(model_path, 'rb') as handle:
                    model = pickle.load(handle)

            elif loader_engine == 'joblib':

                model = joblib.load(model_path)
                
            else:
                raise ValueError('The selected loader_engine is not valid! Please select one of the following: pickle, joblib')

        self.model = model
       
    def predict(self, X: np.array):
        
        # Reshape X data
        _X = X.reshape(-1, X.shape[0])
        _prediction = self.model.predict(_X)[0]
        
        return _prediction
