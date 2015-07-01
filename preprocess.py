# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 09:36:14 2015

@author: LLP-admin
"""
import pandas as pd
#import extract_filter
#########HELPER FUNCTIONS##############
def extract_filter(filter_path):
    """Given a path to the weka's filter file, 
    return a list of selected features."""
    with open(filter_path) as f:
        lnum = 0
        for line in f:
            lnum += 1  #pointer to the next line to read
            if line.strip().startswith('Selected attributes:'):
                print "next line to read: ",lnum
                break
            
        features = []
        for line in f:  #keep reading from where we stopped (from 'break' point)
            if len(line.strip()) != 0:          
                features.append(line.strip())  
        features.append('class')
    return features
                 
############################################
############################################
############################################
############################################

def apply_filter(rfilename):
    """Given the filename in one fixed directory, apply the filter (that selects the best features)
    and get only the data under the selected features."""
    
    dirname = 'C:/cygwin64/home/LLP-admin/GUI4ML/token-experiment-dataset/'
#    rfilename = 'features_0'
#    rfilename = raw_input('Enter the filename (without .csv): ')
    read_path = dirname + rfilename + '.csv'
    df = pd.read_csv(read_path, sep ='\t')
    
    ##Get the list of features to use
    filter_name = 'cfs_filter'
    filter_path = 'C:/cygwin64/home/LLP-admin/GUI4ML/filters/'
    selected_features = extract_filter(filter_path + filter_name)
    #print selected_features
    #
    ###Select ony the 'good' columns
    selected_df = df[selected_features]
    
    ##Write the filtered dataframe as a csv
    out_dirname = dirname +'filtered/'
    ofilename = rfilename + '_' + filter_name
    write_path = out_dirname + ofilename + '.csv'
    selected_df.to_csv(write_path, index = False)

name_list = ['features_'+ str(i) for i in range(0,6)]
apply_filter('features')
for f in name_list:
    apply_filter(f)


