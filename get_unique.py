# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 13:51:19 2015

@author: Hayley Song
"""
import pandas as pd

def get_unique(fname):
    directory = "C:/cygwin64/home/LLP-admin/GUI4ML/token-experiment-dataset/filtered/"
    path = directory + fname + ".csv"
    df = pd.read_csv(path)
    classes = df.ix[:,-1]
    unique_classes = classes.unique()
    print unique_classes.shape

for i in range(0,6):
    fname = "features_" + str(i) +"_cfs_filter"
    get_unique(fname)
    