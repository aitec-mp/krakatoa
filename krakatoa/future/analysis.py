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
import seaborn as sns
from .preprocess import DataClean

from sklearn.decomposition import PCA

# ============================================================


class Analytics(DataClean):

    def __init__(self, target=None):
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

        result = {'type': data_type, 'label': list(x), 'values': list(y)}
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
                try:
                    self.described[col]['dist'] = self.columnDist(col)
                except:
                    self.described[col]['dist'] = {}

        return self.described

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

    def diagnostic(self):
        '''
        Disclaimer!

        This function returns some paremeters like -> 'missing', 'unique', 'target'

        When those returns as False, means that they are not ok and need attention!
        The True flag, means that there are no missing data, the target is solid or there are no unique columns!
        '''

        (rows, cols) = self.dataset.shape
        res_diagnostic = {'missing': True, 'unique': True,
                          'target': True, 'outliers': True, 'rows': rows, 'cols': cols}
        
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
        if self.target is None:
            res_diagnostic['target'] = False
        else:
            if self.target not in set(self.dataset.columns):
                res_diagnostic['target'] = False

        # Check for outliers
        for col in self.dataset.columns:
            try:
                outliers = self.iqrOutliers(column=col)
                if outliers['has_outliers']:
                    res_diagnostic['outliers'] = False
                    break
            except:
                pass

        return res_diagnostic

    def columnInfo(self, column: str, get_values: bool = True, get_values_method: str = 'head',
                   get_values_size: int = 10, most_common_size: int = 2):

        info = {
            'type': str(),
            'values': list(),
            'stats': dict(),
            'outliers': dict()
        }

        info['type'] = str(self.dataset[column].dtype)

        # Implement method to verify column types
        # Getting most common values
        #Workaround, to avoid numpy sort error for mixed types
        if info['type'] == 'object':
            column_values = self.dataset[column].astype(str)
        else:
            column_values = self.dataset[column]

        values, count = np.unique(column_values, return_counts=True)
        count_sort_ind = np.argsort(-count)

        ordered_values = values[count_sort_ind]
        total_size = count.sum()
        perc_count = (count[count_sort_ind]/total_size)*100

        # top - most common values
        perc_sum = 0
        for k, (name, perc) in enumerate(zip(ordered_values, perc_count)):
            # check size
            if k >= most_common_size:
                # Calculate others
                info['stats']['most_common'].append(
                    {'name': 'other', 'perc': 100-round(perc_sum, 2)})
                break

            if 'most_common' not in info['stats'].keys():
                info['stats']['most_common'] = []

            info['stats']['most_common'].append(
                {'name': name, 'perc': round(perc, 2)})

            perc_sum += perc

        # unique values count
        info['stats'].update({'unique': self.dataset[column].nunique()})

        # unique values
        info['stats'].update(
            {'unique_values': list(self.dataset[column].unique())})

        # Missing values
        info['stats'].update(
            {'missing_values': int(self.dataset[column][self.dataset[column] == ''].count())})

        # Null values
        info['stats'].update(
            {'null_values': int(self.dataset[column][self.dataset[column].isnull()].count())})
        
        # Count
        info['stats'].update(
            {'count' : len(self.dataset[column])}
        )

        # Get outliers
        if column in self.numeric_cols:
            info['outliers'] = self.iqrOutliers(column)

            # get describe - updated to include median
            # info['stats'].update(self.dataset[column].describe())
            describe_res = self.dataset[column].describe()

            describe_res['median'] = np.median(self.dataset[column])

            info['stats'].update(describe_res)

        # get column values
        if get_values:
            if get_values_method == 'tail':
                info['values'] = list(
                    self.dataset[column].tail(get_values_size).values)
            else:
                info['values'] = list(
                    self.dataset[column].head(get_values_size).values)

        return info

    def allColumnInfo(self, get_values: bool = True, get_values_method: str = 'head',
                      get_values_size: int = 10, ignore_columns: list = [],
                      most_common_size: int = 2, output_as: str = 'dict'):

        columns = [
            col for col in self.dataset.columns if col not in ignore_columns]

        if output_as == 'dict':
            result = {}
            for col in columns:
                result[col] = self.columnInfo(
                    col, get_values=get_values, get_values_method=get_values_method, get_values_size=get_values_size,
                    most_common_size=most_common_size)

            return result
        elif output_as == 'list':
            result = []
            for col in columns:
                dict_res = self.columnInfo(
                    col, get_values=get_values, get_values_method=get_values_method, get_values_size=get_values_size,
                    most_common_size=most_common_size)
                dict_res.update({'column': col})
                result.append(dict_res)
            return result
        else:
            raise ValueError(
                'The output must be one of the following : dict | list')

    def countPlot(self, column: str, plot: bool = False, **kwargs):

        try:
            # Output value counts
            df = self.dataset[self.dataset[column].notnull()][column]
            count = df.value_counts().to_dict()
            label, values = zip(*count.items())

            # Plot chart
            if plot:
                sns.countplot(x=df, **kwargs)

            return {
                'label': list(label),
                'values': list(values)
            }
        except:
            raise ValueError(
                'Could not count the data values. Check the data type and if the values are valids!')

    def histPlot(self, column: str, plot: bool = False, with_interval: bool = False, interval_type: str = 'list',  **kwargs):
        df = self.dataset[self.dataset[column].notnull()][column]

        y, x = np.histogram(df)

        if with_interval:
            new_x = []

            for n, i in enumerate(x):
                if n > 0:
                    if interval_type == 'list':
                        new_x.append([x[n-1], i])
                        
                    elif interval_type == 'str':
                        new_x.append(f'{x[n-1]} - {i}')
            x = new_x

        if plot:
            sns.histplot(x=df, **kwargs)

        return {
            'label': list(x),
            'values': list(y)
        }

    def boxPlot(self, column: str, plot: bool = False, method: str = 'midpoint', **kwargs):
        df = self.dataset[self.dataset[column].notnull()][column]

        iqr = self.iqrOutliers(column, method=method)

        if plot:
            sns.boxplot(df, **kwargs)

        return iqr

    def pca(self, n_components=1, **kwargs):

        pca = PCA(n_components=n_components, **kwargs)

        pca.fit(self.dataset)

        return pca

