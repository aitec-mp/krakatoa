# -*- coding: utf-8 -*-

'''
Data Evaluation (:mod:`krakatoa.future.evaluate`)
============================================================
'''

#============================================================
# Imports
#============================================================

import pandas as pd
import numpy as np

#============================================================
def categoryPerFeature(df, columns, sort=True):
    n_categories_per_feature = df[columns].apply(lambda x : len(set(x)))
    
    if sort:
        n_categories_per_feature = n_categories_per_feature.sort_values(ascending=False)
        
    
    return n_categories_per_feature
    
