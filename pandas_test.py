# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 16:40:21 2015

@author: LLP-admin
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import *
df = DataFrame({'A' : [0,1,2], 'B' : [1,1,1], 'C' : [2,2,2], 'D' : [0,1,None]})#, 'class': ['a','a','a','b','b','b']})
#print df
#
#
#columns= np.array(df.columns)
#print "np'std: \n", np.std(df)
#isNotConstant = (np.std(df) != 0)
#notConstantCols= [df.columns[i] for (i,v) in enumerate(np.std(df)) if v != 0]
#filtered_df = df[notConstantCols]
#print "filtered: \n", filtered_df
