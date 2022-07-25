# -*- coding: utf-8 -*-

'''
Data preprocessing (:mod:`krakatoa.future.preprocess`)
============================================================
'''

#============================================================
# Imports
#============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#============================================================
def changeColType(df, columns, newType):
    
    for col in columns:
        df[col] = df[col].astype(newType)
        
    return df

def splitDataset(x, y, test_size=0.3, random_state=0):
    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
    
    return x_train, x_test, y_train, y_test

def scale(dataset, columns, scaler='min_max'):
    
    scalers = {
        'min_max' : MinMaxScaler(),
        'max_abs' : MaxAbsScaler(),
        'robust' : RobustScaler()
        }
    
    cur_scaler = scalers.get(scaler)
    dataset[columns] = cur_scaler.fit_transform(dataset[columns])
    
    return dataset
    
    
    
    
