# -*- coding: utf-8 -*-

'''
Data Analysis (:mod:`krakatoa.future.analysis`)
============================================================
'''

#============================================================
# Imports
#============================================================

import pandas as pd
import numpy as np
from .preprocess import DataClean

#============================================================
class Analytics(DataClean):

    def __init__(self):
        super().__init__()
        pass

    def loadDataset(self, dataset, load_from="dataframe"):
        if load_from == "dataframe":
            self.dataset = dataset
            self.originalDataset = dataset
        elif load_from == "dict":
            self.dataset = pd.DataFrame(dataset)
            self.originalDataset = pd.DataFrame(dataset)
        else:
            print("Error! Select the right Dataset type and set it to load_from variable ('dataframe', 'dict')")


    def columnDist(self, column):

        # Verify if column is numeric or string
        # When numeric and unique values > min numer (eg: 50) histogram can be used
        # When string alwais use a count plots

        dtype = str(self.dataset[column].dtype)

        if dtype in ['object', 'category', 'string']:
            data_type = "count"
            values = self.dataset[column].value_counts()
            x = list(values.keys().astype("string"))
            y = list(values.values)

        else:          
            # Check the amount of unique values
            percUnique = (self.dataset[column].nunique() / self.dataset.shape[0]) * 100
            countUnique = self.dataset[column].nunique()

            if percUnique > 10 or countUnique > 10: #TODO revisitar esse percentual
                data_type = "histogram"
                y, x = np.histogram(self.dataset[column])

            else:
                data_type = "count"
                values = self.dataset[column].value_counts()
                x = list(values.keys().astype("string"))
                y = list(values.values)

        result = {'data_type': data_type, 'x' : list(x), 'y' : list(y)}
        return result

    def targetDist(self, target):

        result = self.countColumnValues(target)

        self.distTarget = result

        return self.distTarget

    def checkTypes(self, threshold=90, changeDtypes=False):

        # Search for unique features and column types
        super().getColType()
        super()._getUniqueFeatures()

        textColumns = []
        catColumns = []

        # Check if any of the features represents more than threshold (default 90%)
        # If is text object, will help to identify if is category or text
        # when is above 90% generally is a text, otherwise is category
        unique = self.uniquePerc[self.uniquePerc >= threshold]
        if unique.shape[0] > 0:

            # Select category cols
            textColumns = list(unique[unique.keys().isin(self.category_cols)].keys())

        catColumns = [x for x in self.category_cols if x not in textColumns]

        self.textColumns = textColumns
        self.catColumns = catColumns

        # Converte o dtype das colunas de acordo com o que foi identificado
        if changeDtypes:
            conversion_dict = {
                "category" : self.catColumns,
                "string" : self.textColumns
            }

            for dtype, columns in conversion_dict.items():
                for col in columns:
                    self.dataset[col] = self.dataset[col].astype(dtype)

    def describe(self, dataset=None, load_from="dataframe", get_dist=False):

        if dataset != None:
            self.loadDataset(dataset=dataset, load_from=load_from)

        # Execute type checker function
        self.checkTypes(threshold=90, changeDtypes=True)

        # Describe with pandas
        pdDescribe = self.dataset.describe(percentiles=[], include="all")
        pdDescribe.fillna("", inplace=True)

        # Get columns types and add to describe dataframe
        columns_dtypes = list(self.dataset.dtypes.keys())
        values_dtype = list(self.dataset.dtypes.astype("string").values)
        dfDtypes = pd.DataFrame(data=[values_dtype], columns=columns_dtypes, index=['dtype'])

        # Add another unique counter
        columns_unique = list(self.uniqueCount.keys())
        values_unique = list(self.uniqueCount.astype("string").values)
        dfUnique = pd.DataFrame(data=[values_unique], columns=columns_unique, index=['nunique'])

        described = pd.concat([pdDescribe, dfDtypes, dfUnique])

        # Transform to dict
        self.described = described.to_dict()

        if get_dist:
            for col in described.keys():
                
                self.described[col]['dist'] = self.columnDist(col)

        return self.described
            
        