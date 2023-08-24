# -*- coding: utf-8 -*-

'''
Configuration file(:mod:`krakatoa._getmodels`)
============================================================
'''

# ============================================================
# Imports
# ============================================================

from ._config import getConfigRegr, getConfigClass

# ============================================================
# Get models
# ============================================================
def getModels(mode, modelClasses, selMode = 'type', random_state=0):
    

    if mode == 'regression':
        config = getConfigRegr()
    elif mode == 'classification':
        config = getConfigClass()
    else:
        #TODO error implementation
        print('No configuration was selected')
        config = None

    if config != None:

        configVals = [x for x in config.values()] 

        result = []

        if selMode in ['type', 'name']:
            
            for m in modelClasses:
                res = list(filter(lambda x: x[selMode] == m, configVals))
                
                for r in res:
                    result.append(r)
        
        return result
