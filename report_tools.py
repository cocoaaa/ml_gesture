# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 10:52:35 2015

@author: LLP-admin
"""
import os
import numpy as np
import pandas as p


def build_num2features(FS_report):
    """
    Build a dictionary with key = number of best features selected and value = %accuracy.
    EX) {1: ["mean"], 2: ["mean","var"], 3:["var", "angle","center"],...etc}
    """
    newDict = {};
    for fset in FS_report.iterkeys():
        newDict[len(fset)] = list(set(fset))
    return newDict
    
    
def build_sortedReportList(FS_report):
    """
    [(1,%), (2,%), (3,%), ... ]
    
    """
    temp = [(len(k), FS_report[k]) for k in FS_report.iterkeys() if type(k) == frozenset]
    return temp.sort( key = lambda t: t[0])
          
    
def findFirstNPeaks(report, N=3):
    """
    Given the report dictionary whose key is frozenset(a list of features) and value is %accuracy,
    return the list of the first N peaks: [ (x_1, %accuracy), ... (x_n, %accuracy) ].
    Assumes that the report starts at key = 1 (i.e. training session index is based on 1)
    """
    #First build a dictionary with key = nFeatures and value = %accuracy
    
    num_report = {}
    for k in report.iterkeys():
        num_report[len(k)] = report[k];
       
    
    #Add boundaries to the report
    num_report[0] = -float('inf') #Any value <0 will do
    num_report[len(report)+1] = -float('inf')
    
    
    peaks = [];
    #Do search starting at the smaller index
    for i in range(1, len(report) +1):
        if len(peaks) >= N:
            break
        if (num_report[i-1] <= num_report[i]) and (num_report[i] >= num_report[i+1]):
            peaks.append( (i, num_report[i]) )
            
    return peaks #No peak found. This should never happen. 
           

def suggestFeatures(FS_report, fname, clfName, toWrite = True):
    """
    Take a look at the first three peaks and suggest the nFeatures with the 
    highest resulting accuracy.
    Outputs (nFeatures, a list of selected feature names) to outPath, if toSave (default)
    """
    three_peaks = findFirstNPeaks(FS_report, 3);
    num2features = build_num2features(FS_report);
    
    best_num = max(iter(three_peaks), key = lambda x: x[1])[0]
    best_features = num2features[best_num]
    
    if toWrite:
        outDir = '..\\feature_suggestion\\' + clfName+ '\\'
        outName = fname +'.txt'
        outPath = os.path.join(outDir, outName)
    
         #check if the outpath is already created.
        try:
            os.makedirs(outDir);
        except OSError:
            if not os.path.isdir(outDir):
                raise
        
        with open(outPath, 'w') as f:
            f.write("\n---------------Feature Suggestion---------------")
            f.write("\n\tBased on the ranking by Random Forest, we suggest the following:")
            f.write("\n1. Number of features: " + str(best_num))
            f.write("\n2. Features: " + str(best_features) )
            f.write("\n\n----------End of feature recommendation----------")
            f.close()
            
    return (best_num, best_features)
    
###############test##########################

    
    