# -*- coding: utf-8 -*-

'''
Configuration file(:mod:`krakatoa._getmodels`)
============================================================
'''

# ============================================================
# Imports
# ============================================================

from ._config import getConfig

# ============================================================
# Get models
# ============================================================
def getModels(modelClasses, selMode = 'type', random_state=0):
    
    config = getConfig()
    configVals = [x for x in config.values()] 
    result = []

    if selMode in ['type', 'name']:
        
        for m in modelClasses:
            res = list(filter(lambda x: x[selMode] == m, configVals))
            
            for r in res:
                result.append(r)
    
    return result
