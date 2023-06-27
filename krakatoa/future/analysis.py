# -*- coding: utf-8 -*-

'''
Data Analysis (:mod:`krakatoa.future.analysis`)
============================================================
'''

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from .preprocess import DataClean

# ============================================================


class Analytics(DataClean):

    def __init__(self, target):
        super().__init__(target)
        pass

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
            percUnique = (
                self.dataset[column].nunique() / self.dataset.shape[0]) * 100
            countUnique = self.dataset[column].nunique()

            if percUnique > 10 or countUnique > 10:  # TODO revisitar esse percentual
                data_type = "histogram"
                # y = freq | x = edges
                y, x = np.histogram(self.dataset[column])
                '''
                Chart implementation ex:
                ax.bar(x[:-1], y, width=np.diff(x), edgecolor="black", align="edge")
                '''

            else:
                data_type = "count"
                values = self.dataset[column].value_counts()
                x = list(values.keys().astype("string"))
                y = list(values.values)

        result = {'data_type': data_type, 'x': list(x), 'y': list(y)}
        return result

    def targetDist(self, target):

        result = self.columnDist(target)

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
            textColumns = list(
                unique[unique.keys().isin(self.category_cols)].keys())

        catColumns = [x for x in self.category_cols if x not in textColumns]

        self.textColumns = textColumns
        self.catColumns = catColumns

        # Converte o dtype das colunas de acordo com o que foi identificado
        if changeDtypes:
            conversion_dict = {
                "category": self.catColumns,
                "string": self.textColumns
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
        dfDtypes = pd.DataFrame(
            data=[values_dtype], columns=columns_dtypes, index=['dtype'])

        # Add another unique counter
        columns_unique = list(self.uniqueCount.keys())
        values_unique = list(self.uniqueCount.astype("string").values)
        dfUnique = pd.DataFrame(
            data=[values_unique], columns=columns_unique, index=['nunique'])

        described = pd.concat([pdDescribe, dfDtypes, dfUnique])

        # Transform to dict
        self.described = described.to_dict()

        if get_dist:
            for col in described.keys():

                self.described[col]['dist'] = self.columnDist(col)

        return self.described

    def diagnostic(self):
        '''
        Disclaimer!

        This function returns some paremeters like -> 'missing', 'unique', 'target'

        When those returns as False, means that they are not ok and need attention!
        The True flag, means that there are no missing data, the target is solid or there are no unique columns!
        '''

        res_diagnostic = {'missing': True, 'unique': True, 'target': True}
        # Check for missing data
        null_data = self._nullPercFeatures()
        null_diagnostic = null_data[null_data['count'] > 0][[
            'variable', 'count']].to_dict('records')

        if len(null_diagnostic) > 0:
            null_diagnostic_items = {}
            for item in null_diagnostic:
                null_diagnostic_items[item['variable']] = item['count']
            res_diagnostic['missing'] = False
            res_diagnostic['missing_data'] = null_diagnostic_items

        # Check for unique data
        unique_data = self.dataset.nunique()
        unique_diagnostic = unique_data[unique_data == 1].to_dict()

        if len(unique_diagnostic.keys()) > 1:
            res_diagnostic['unique'] = False
            res_diagnostic['unique_data'] = unique_diagnostic

        # Check for valid target
        if not self.target:
            res_diagnostic['target'] = False
        else:
            if self.target not in set(self.dataset.columns):
                res_diagnostic['target'] = False

        return res_diagnostic

    def iqrOutliers(self, column: str, method: str = 'midpoint'):

        if column in self.numeric_cols:
            Q1 = np.percentile(self.dataset[column], 25, method=method)
            Q2 = np.percentile(self.dataset[column], 50, method=method)
            Q3 = np.percentile(self.dataset[column], 75, method=method)
            Q4 = np.percentile(self.dataset[column], 100, method=method)
            IQR = Q3 - Q1

            # Calculate bounds
            lower_bound = Q1-1.5*IQR
            rows_lower_bound = list(
                self.dataset[self.dataset[column] <= lower_bound].index)

            upper_bound = Q3+1.5*IQR
            rows_upper_bound = list(
                self.dataset[self.dataset[column] >= upper_bound].index)

            has_outliers = False if len(
                rows_lower_bound) + len(rows_upper_bound) == 0 else True
            return {
                'has_outliers': has_outliers,
                'lower_bound': lower_bound,
                'rows_lower_bound': rows_lower_bound,
                'upper_bound': upper_bound,
                'rows_upper_bound': rows_upper_bound,
                'Q1': Q1,
                'Q2': Q2,
                'Q3': Q3,
                'Q4': Q4,
                'IQR': IQR
            }

        else:
            raise ValueError(
                'Outliers calculations can be performed only on numeric columns!')

    def searchOutliers(self, method: str = 'midpoint'):
        outliers = {}
        for col in self.numeric_cols:

            iqr_outliers = self.iqrOutliers(col, method)

            if iqr_outliers['has_outliers']:
                outliers[col] = iqr_outliers

        return outliers

    def columnInfo(self, column: str, get_values: bool = True, get_values_method: str = 'head', get_values_size: int = 10):

        info = {
            'type': str(),
            'values': list(),
            'stats': dict(),
            'outliers': dict()
        }

        info['type'] = str(self.dataset[column].dtype)

        # Implement method to verify column types
        # Getting most common values
        values, count = np.unique(self.dataset[column], return_counts=True)
        count_sort_ind = np.argsort(-count)

        ordered_values = values[count_sort_ind]
        total_size = count.sum()
        perc_count = (count[count_sort_ind]/total_size)*100

        # top 10 most common
        for name, perc in zip(ordered_values, perc_count):
            if 'most_common' not in info['stats'].keys():
                info['stats']['most_common'] = []

            info['stats']['most_common'].append(
                {'name': name, 'perc': round(perc, 2)})

        # unique values count
        info['stats'].update({'unique': self.dataset[column].nunique()})

        # unique values
        info['stats'].update(
            {'unique_values': list(self.dataset[column].unique())})

        # Missing values
        info['stats'].update(
            {'missing_values': self.dataset[column][self.dataset[column] == ''].count()})

        # Null values
        info['stats'].update(
            {'null_values': self.dataset[column][self.dataset[column].isnull()].count()})

        # Get outliers
        if column in self.numeric_cols:
            info['outliers'] = self.iqrOutliers(column)

            # get describe
            info['stats'].update(self.dataset[column].describe())

        # get column values
        if get_values:
            if get_values_method == 'tail':
                info['values'] = list(
                    self.dataset[column].tail(get_values_size).values)
            else:
                info['values'] = list(
                    self.dataset[column].head(get_values_size).values)
                
        return info

    def allColumnInfo(self, get_values: bool = True, get_values_method: str = 'head', get_values_size: int = 10, ignore_columns: list = [], output_as: str = 'dict'):

        columns = [
            col for col in self.dataset.columns if col not in ignore_columns]

       
        if output_as == 'dict':
            result = {}
            for col in columns:
                result[col] = self.columnInfo(
                    col, get_values=get_values, get_values_method=get_values_method, get_values_size=get_values_size)
                
            return result
        elif output_as == 'list':
            result = []
            for col in columns:
                dict_res = self.columnInfo(
                    col, get_values=get_values, get_values_method=get_values_method, get_values_size=get_values_size)
                dict_res.update({'column' : col})
                result.append(dict_res)
            return result
        else:
            raise ValueError('The output must be one of the following : dict | list')

