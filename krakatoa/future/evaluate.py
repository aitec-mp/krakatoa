# -*- coding: utf-8 -*-

'''
Data Evaluation (:mod:`krakatoa.future.evaluate`)
============================================================
'''

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np

# ============================================================


def categoryPerFeature(df, columns, sort=True):
    n_categories_per_feature = df[columns].apply(lambda x: len(set(x)))

    if sort:
        n_categories_per_feature = n_categories_per_feature.sort_values(
            ascending=False)

    return n_categories_per_feature


def countNull(df):

    # result = ''

    # if mode == 'perc':
    #     result = (((df.isnull().sum() | df.eq('').sum())/df.shape[0])*100)
    # elif mode == 'count':
    #     result = (df.isnull().sum() | df.eq('').sum()).to_dict()

    return pd.DataFrame({
        'variable': list(df.columns),
        'perc': (((df.isnull().sum() | df.eq('').sum())/df.shape[0])*100).values,
        'count': (df.isnull().sum() | df.eq('').sum()).values
    })
