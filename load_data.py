# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:26:09 2015

@author: LLP-admin
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

def divideByClass(filepath, toSave = True):
    """
    Input: a complete filepath
    toSave (bool parameter): if true, save each class file to current directory.
    It can handle delimiter of ',' or '\t' (i.e. tab).
    The file to be read must have 'class' feature. 
    Divide the dataset by class and save each class's dataset into a csv file named by the class value,
    in the same directory. 
    Returns 0 if successful divide and save.
    """
#    print "current filepath: ", filepath;
    try: 
        df = pd.io.parsers.read_csv(filepath,delimiter=','); 
        class_column = df['class'];
        
    except KeyError:
        df = pd.io.parsers.read_csv(filepath, delimiter='\t'); 
        class_column = df['class'];
        
    #print an error message and then re-raise the exception 
    #(allowing a caller to handle the exception as well):
    except:
        print "Unexpected error:", sys.exc_info()[0];
        raise;

    class_values = class_column.unique();
    
    #Initialize a dictionary whose key is class value and value is dataframe of the class value.
    dict_df_class = {};
    for (i, value) in enumerate(class_values):
        #write to a csv file
        #Extract the directory path from the filepath.
        #Be careful since the filepath might end with '\'.
        splitAt = filepath.rfind('\\',0,len(filepath)-3); 
        dirpath = filepath[:splitAt +1];
        outpath = dirpath + class_values[i] + ".csv"; 
        
        df_class = df[df['class']==value];
        dict_df_class[class_values[i]] = (df_class);
        
        #If toSave is set to true, we save each file 
        if toSave:
#           print "\nNow, saving ", class_values[i], "....."; 
            df_class.to_csv(outpath, index = False);
#           print "Done saving this class value dataframe";
        
    #Return the dictionary after the successful save.    
    return dict_df_class; 


def splitXY(df):
    """Given a dataFrame of data, 
    split it into X (features and values) and Y (label vector).
    Assume the last column is the class column.
    """
    return df[df.columns[0:-1]], df[df.columns[-1]];
    
def piped_standardize(train_x, test_x):
    """
    Input: Pandas DataFrame with all numeric values.  It does not have the label feature.
    Each column is a feature, each row is an instance (example).
    Returns the standardized train_x (as a data frame), and the test (data frame) standardized based on the train data's mean and std.
    """
    header = train_x.columns
    scaler = StandardScaler().fit(train_x);
#    std_train_x = scaler.transform(train_x);
#    std_test_x = scaler.transform(test_x);
    return ( pd.DataFrame(scaler.transform(train_x), columns = header), pd.DataFrame(scaler.transform(test_x), columns = header));
    
    
def findFirstPeak(report):
    """Given the report dictionary whose key is number of batch training vs %accuracy,
    return the first peak: (firstPeak_x, %accuracy)
    report starts at key = 1 (i.e. training session index is based on 1)
    """
    #Add boundaries to the report
    report[0] = -float('inf') #Any value <0 will do
    report[len(report)+1] = -float('inf')
    
    #Do search starting at the smaller index
    for i in range(1, len(report) +1):
        if (report[i-1] <=report[i]) and (report[i] >= report[i+1]):
            return (i, float("{0:.2f}".format(report[i])) )
    return None #No peak found. This should never happen. 
           

    