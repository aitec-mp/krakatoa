# -*- coding: utf-8 -*-

'''
Experiment creation (:mod:`krakatoa.future.experiment`)
============================================================
'''

# ============================================================
# Imports
# ============================================================

import pandas as pd

from ..models.quick import Regressor  # Quick models regressor
from ..models.autotune import RegressorAF  # Auto models regressor
from .preprocess import DataClean, splitDataset

# ============================================================


class Experiment(DataClean):

    def __init__(self):
        super().__init__()
        self.dataset = pd.DataFrame()
        self.target = None

    def _setTarget(self, target):
        self.target = target

    def create(self, dataset, target=None):

        # set target if informed
        if target != None:
            self._setTarget(target)

        # datasetClass = DataClean()

        # Fast and simple method
        # datasetClass.fit_transform(dataset)
        super().fit_transform(dataset)

    def train(self, method='quick', estimator=['linear', 'tree'], **kwargs):

        X, y = super().splitTrainTest(self.target)

        if method == 'quick':
            regressor = Regressor()
            result = regressor.multiRegression(
                x=X, y=y, models=estimator, **kwargs)
        elif method == 'auto':
            regressor = RegressorAF()
            result = regressor.fit(estimator=estimator, x=X, y=y, **kwargs)

        return result
