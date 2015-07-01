# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 15:59:37 2015

@author: LLP-admin
"""

def extract_filter(filter_path):
    """Given a path to the weka's filter file, 
    return a list of selected features."""
    with open(filepath) as f:
        lnum = 0
        for line in f:
            lnum += 1  #pointer to the next line to read
            if line.strip().startswith('Selected attributes:'):
                print "next line to read: ",lnum
                break
            
        features = []
        for line in f:  #keep reading from where we stopped (from 'break' point)
#            if len(line.strip()) != 0:          
            features.append(line.strip())   
        return features
                 
            
#test
#filter_path = 'C:/cygwin64/home/LLP-admin/GUI4ML/filters/cfs_filter'
#extract_filter(filter_path)