# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:02:05 2015

@author: LLP-admin
"""
import numpy as np 
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif


def isNotConstantCol(rawTrain_x):
    """
    Inputs: 
    -data_x: data with only features as Pandas dataframe    
    Returns: the list of column names whose values change (i.e. the std of the feature colm is not zero).
    
    """
    return np.std(rawTrain_x) != 0


#def isSelectedCol():
    
#####Test#################################
##############################################################################
##############################################################################
#dirpath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\'
#fpath = dirpath +"features_1.csv";
#x = pd.DataFrame({'A' : [0,1,2], 'B' : [1,1,1], 'C' : [2,2,2], 'D' : [0,1,3]})
#y = pd.DataFrame({'class': ['a','b','b']})
#print isNotConstantCol(x)
#a= isKBestCol(x,y, k =3)