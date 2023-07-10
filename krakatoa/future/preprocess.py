# -*- coding: utf-8 -*-

'''
Data preprocessing (:mod:`krakatoa.future.preprocess`)
============================================================
'''

#============================================================
# Imports
#============================================================

import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from krakatoa.future.evaluate import countNull
from sklearn.preprocessing import OneHotEncoder

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
        'robust' : RobustScaler(),
        'standard' : StandardScaler()
        }
    
    cur_scaler = scalers.get(scaler)
    dataset[columns] = cur_scaler.fit_transform(dataset[columns])
    
    return {
        "scaler" : cur_scaler,
        "dataset" : dataset
    }


# Class designed to dataset management   
class DataClean():

    def __init__(self, target=None):
        self.dataset = pd.DataFrame()
        self.originalDataset = pd.DataFrame()
        self.target = target
    
    def _checkTarget(self):
        if self.target is None:
            raise ValueError('Target must be set in order to execute that function')

        return True

    def loadDataset(self, dataset, load_from="dataframe"):
        if load_from == "dataframe":
            self.dataset = dataset
            self.originalDataset = dataset
        elif load_from == "dict":
            self.dataset = pd.DataFrame(dataset)
            self.originalDataset = pd.DataFrame(dataset)
        else:
            print("Error! Select the right Dataset type and set it to load_from variable ('dataframe', 'dict')")
        self.getColType()
        self._getUniqueFeatures()
        self._nullPercFeatures()

    def getColType(self):
        
        # TODO precisamos ainda identificar colunas de outros tipos , como data e quando é numerica e categorica
        catCols = []
        numCols = []
        datCols = []
        floatCols = []
        intCols = []
        otherCols = []

        for k, v in self.dataset.dtypes.items():
            # Check for non numeric
            if v in ['object', 'category']:
                # Try to set datetime data
                try:
                    # self.dataset[k] = pd.to_datetime(self.dataset[k], format='%d-%m-%Y %H:%M:%S.%f')
                    self.dataset[k] = pd.to_datetime(self.dataset[k], infer_datetime_format=True)
                    datCols.append(k)
                except:
                    pass
                    catCols.append(k)
            else:
                if v in ['float64', 'float32']:
                    floatCols.append(k)
                    numCols.append(k)
                elif v in ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']:
                    intCols.append(k)
                    numCols.append(k)
                else:
                    otherCols.append(k)
                
        self.category_cols = catCols
        self.numeric_cols = numCols
        self.integer_cols = intCols
        self.float_cols = floatCols
        self.other_cols = otherCols
        self.datetime_cols = datCols
        if self.target is not None:
            self.target_col = [self.target]
        else:
            self.target_col = []

    def splitTrainTest(self):

        self._checkTarget()

        X = self.dataset.drop(columns=[self.target])
        y = self.dataset[self.target]

        self.X = X
        self.y = y

        return X, y

    def _nullPercFeatures(self):
        result = countNull(self.dataset)

        self.percNull = result
        return result

    def _getUniqueFeatures(self):
        dataset = self.dataset.copy()

        catUniqueCount = dataset[self.category_cols].nunique()
        catUniquePerc = (dataset[self.category_cols].nunique() / dataset.shape[0])*100
        uniqueCount = dataset.nunique()
        uniquePerc = (dataset.nunique() / dataset.shape[0])*100

        self.catUniqueCount = catUniqueCount
        self.catUniquePerc = catUniquePerc
        self.uniqueCount = uniqueCount
        self.uniquePerc = uniquePerc

    def dropColumns(self, columns):

        self.dataset.drop(columns=columns, inplace=True)
        return self.dataset
    
    def dropNaColumns(self, threshold=50):
        
        percNull = self._nullPercFeatures()

        self.dataset.drop(columns=list(percNull[percNull > threshold].keys()), inplace=True)
        return self.dataset
    
    def fillNa(self, strategy='mean'): #mean / most frequent | drop 
        # preencher com media | mais frequente (caso categorico)
        self.getColType()
        dataset = self.dataset.copy()
        if strategy == "mean":
            for numc in self.numeric_cols:
                mean = dataset[numc].mean()
                dataset[numc].fillna(mean, inplace=True)
            
            # categoricos por mais frequentes
            for catc in self.category_cols:
                mode = dataset[catc].mode()[0]
                dataset[catc].fillna(mode, inplace=True)
        elif strategy == "drop":
            dataset.dropna(inplace=True)

        self.dataset = dataset
        return self.dataset

    def cleanUnique(self, threshold=500):
        
        self._checkTarget()
        self._getUniqueFeatures()

        dataset = self.dataset.copy()
        col_drop = list(self.uniqueCount[self.uniqueCount > threshold].keys())

        # Dont drop target column for any reason | Dont drop numeric columns
        col_drop = [x for x in col_drop if x != self.target and x not in (self.numeric_cols)] 

        dataset.drop(columns=col_drop, inplace=True)

        self.dataset = dataset


        # refresh column data in self
        self.getColType()

        return self.dataset

    def getDummies(self):

        # self.dataset = pd.get_dummies(self.dataset)
        # return self.dataset
        self.dataset.reset_index(inplace=True, drop=True)

        cat_cols = self.category_cols

        # Instancia e fit one hot encoder
        hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        hot.fit(self.dataset[cat_cols])

        hot_ds = hot.transform(self.dataset[cat_cols])

        # Cria dataframe transformada
        hot_df = pd.DataFrame(hot_ds, columns = hot.get_feature_names_out(input_features=cat_cols))

        # Concatena o dataframe do one hot com o original
        concat_df = pd.concat([self.dataset, hot_df], axis=1).drop(columns=cat_cols, axis=1)
        self.oneHotEncoding = hot
        self.dataset = concat_df

        return self.dataset

    def dataScale(self, scaler='min_max'):

        res_scale = scale(self.dataset, columns=self.numeric_cols, scaler=scaler)

        self.dataset = res_scale["dataset"]
        self.scaler = res_scale["scaler"]

        return self.dataset

    def pipeline(self, steps):

        _config = {
            'dataset' : self.loadDataset,
            'splitColType' : self.getColType,
            'dropNaCol' : self.dropNaColumns,
            'fillNa' : self.fillNa,
            'cleanUnique' : self.cleanUnique,
            'getDummies' : self.getDummies,
            'scale' : self.dataScale,
            'dropCol' : self.dropColumns,
            # 'target' : self.setTarget,
        }

        for step, args in steps:
            fun = _config.get(step) #Else mostra error TODO
            fun(**args)
            
        return self.dataset

    def fit_transform(self, dataset, dropThresh=50, fillNaStrategy='mean', uniqueThresh=500, scaler='min_max'): #Metodo automatizado | pipeline

        # self.loadDataset(dataset)
        # self.getColType()
        # self.dropNaColumns(threshold=dropThresh)
        # self.fillNa(strategy=fillNaStrategy)
        # self.cleanUnique(threshold=uniqueThresh)
        # self.getDummies()
        # self.scale(scaler=scaler)

        self.pipeline(steps=[
            ('dataset', {'dataset' : dataset}),
            ('splitColType', {}),
            ('dropNaCol', {'threshold' : dropThresh}),
            ('fillNa', {'strategy' : fillNaStrategy}),
            ('cleanUnique', {'threshold' : uniqueThresh}),
            ('getDummies', {}),
            ('scale', {'scaler' : scaler})
            ])
        
        return self.dataset

    def setColumnType(self, col:str, col_type:str):

        try:
            self.dataset[col] = self.dataset[col].astype(col_type)
            self.getColType()
            return True
        except Exception as e:
            print(e)
            return False
       
    def setTarget(self, target):
        self.target = target

    def consistFormat(self, target='mongodb'):

        if target == 'mongodb':

            # Check for NaT values in 
            for col in self.datetime_cols:
                if self.dataset[col].isnull().sum() > 0:
                    self.dataset[col] = self.dataset[col].astype('object').where(self.dataset[col].notnull(), None)

            # Transform dataframe to list and records
            records = self.dataset.to_dict('records')

            return records
        
        else:
            raise ValueError('Not implemented!')
