# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:06:39 2015

@author: LLP-admin
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
    header=None,
    usecols=[0,1,2]
    )

df.columns=['Class label', 'Alcohol', 'Malic acid']
scaler = preprocessing.StandardScaler().fit(df);
print "mean: ", scaler.mean_
print "std: ", scaler.std_
print "df head:\n ", df.head()
df_std= scaler.transform(df) 
print "\ndf after std \n", pd.DataFrame(df_std, columns = df.columns).head()