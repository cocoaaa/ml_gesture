# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:55:05 2015

@author: LLP-admin
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#encoding
from sklearn.preprocessing import LabelEncoder

from load_data import (divideByClass, splitXY, piped_standardize)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


from itertools import chain, combinations

def g_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    """Returns the generator for powerset of the interable"""
    
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in xrange(len(s)+1))


######################################################################
########################################################################
#memo = {};
def getPwrSet(L):
    """
    Given a list, return a list of all possible subsets (sublists).
    For example, given L = [1,2,3], it returns [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]].
    This algorithm is memoized.  Don't forget the memo (a dictionary) right above the definition
    of this function.
    """
    if frozenset(L) in memo:
        pwrSet = memo[frozenset(L)];
    else:
        #Base case: empty set
        if len(L) == 0:
            print "this should be printed only once if memo is working"
            pwrSet = [L];
        else:
            last_ele = L[-1];
            prevSet = getPwrSet(L[0:-1])
            newSet = [ele + [last_ele] for ele in prevSet];
            pwrSet = prevSet + newSet;
            
        memo[frozenset(L)] = pwrSet;   
#    print 'Done creating powerSets...'
    return pwrSet
###Test for getPwrSet#####
#lists = [ [1], [2], [1,2], [1,3], [1,2,3]];
#L = ['A','B','C']
#print getPwrSet(L)
#print '\nlength: ', len(getPwrSet(L))
######################################################################
######################################################################
    

def makeStdDataSets2(filepath, nTr = 3, nTest = 10):
    """
    Inputs
        -nTr: number of training sessions, i.e. number of training instancers per class.
        -nTest: number of test instances per class
    Returns
        -standardized train_x, test_x and label-encoded (from classes to integers) train_y, test_y  
    Note: test_set is constructed by instances that follow directly the training instances.
    """    
    dict_dataPerClass = divideByClass(filepath);
    sampleDF = dict_dataPerClass.values()[0];
    columns = sampleDF.columns;
    batch =  pd.DataFrame(columns = columns)
    test_set = pd.DataFrame(columns = columns);
        
    for dataPerClass in dict_dataPerClass.itervalues():
#            assert( not(dataPerClass.isnull().any().any())  ) ; print 'No None in this class dataset!'
        batch = batch.append(dataPerClass.iloc[0:nTr]);
        #Now, need to prepare the test data set.
        test_set = test_set.append( dataPerClass.iloc[nTr:nTr+nTest] )
    
     #split the batch into features and labels
    batch_x, train_y = splitXY(batch)
    rawTest_x, rawTest_y = splitXY(test_set)
    #Done creating training and test data sets for this session.
        
    #Standardize the train data.  Apply the mean and std parameter to scale the test data accordingly.
    train_x, test_x = piped_standardize(batch_x, rawTest_x);

    #Make sure the number of features in train_x and test_x are same
    assert(len(train_x.columns) == len(test_x.columns));    
    
    #Label encoding
    # batch_y.index = range(0, len(batch_y))
    le = LabelEncoder()
    le.fit(train_y);
        
    train_y = le.transform(train_y)
    test_y = le.transform(rawTest_y)    
    
    return train_x, train_y, test_x, test_y


def selectFeatureSet_RF(data_x, data_y, nFeatures):
    """Use Random Forest to find the best numFeatures of features, based on the given data_x."""

    rf_filter = RandomForestClassifier(max_features = 'auto')
    rf_filter.fit(data_x, data_y);
    rankings = rf_filter.feature_importances_;
    selectedBool = np.argsort(rankings)[-nFeatures:]
#    selectedBool = sorted(range(len(rankings)), key = lambda x: rankings[x])[-nFeatures:];
    return data_x.columns[selectedBool]    



def evalFeatureSet(train_x, train_y, test_x, test_y, selectedFeatures, classifier):
    if len(selectedFeatures) == 0:
        score = 0.0
        
    train_x = train_x[selectedFeatures];
    test_x = test_x[selectedFeatures];
        
    #Don't need to modify even after more filtering is applied later
    #train the classifier on this batch
    clf = classifier;
    clf.fit(train_x, train_y);

    #test the classifier on the fixed test set
    score = clf.score(test_x, test_y);
    return (frozenset(selectedFeatures), score)
    
    
    
    
def get_FS_report(filepath, classifier, nTr = 3, nTest = 10):
    """Get the report of featureSet size vs. %accuracy using Random Forest as the feature selection filter."""
    #1. Get standardized train and test data
    train_x, train_y, test_x, test_y = makeStdDataSets2(filepath, nTr, nTest);
    
    #Total number of features is the number of columns in train_x ( this should equal that of test_x)
    _, total = train_x.shape
#    2. select features with varying number of features
    FS_report = {};
    for nFeatures in range(1, total +1):
        selectedFeatures = selectFeatureSet_RF(train_x, train_y, nFeatures);
        featureSet, score = evalFeatureSet(train_x, train_y, test_x, test_y, selectedFeatures, classifier)
        FS_report[featureSet] = score;
#        print "\nfeature SET: ", featureSet
#        print "score: ", score
    
    return FS_report    


def plot_FS_report(FS_report, clfName, fname):
    plt.figure();
    plt.xlim([0,24]);plt.xticks(np.arange(0, 24, 1.0));
    plt.ylim([0,1.0]);plt.yticks(np.arange(0, 1.0, 0.1));
    plt.xlabel("number of best features selected")
    plt.ylabel("% accuracy")
    plt.title("Report on: "+ fname+ \
        "\nClassifier: "+ clfName);
    
    for k,v in FS_report.iteritems():
        plt.plot(len(k),v, 'bo')
        plt.hold
        
    plt.show()



def get_PCA_FS_report(filepath, classifier, nTr = 3, nTest = 10):
    """Get the report of featureSet size vs. %accuracy using Random Forest as the feature selection filter.
    PCA is applied after feature selection"""
    #1. Get standardized train and test data
    all_train_x, train_y, all_test_x, test_y = makeStdDataSets2(filepath, nTr, nTest);
    
    #Total number of features is the number of columns in train_x ( this should equal that of test_x)
    _, total = all_train_x.shape
#    2. select features with varying number of features
    PCA_report = {};
    for nFeatures in range(1, total +1):
        
        selectedFeatures = selectFeatureSet_RF(all_train_x, train_y, nFeatures);
#        print selectedFeatures
        
#        
        #Select only the top-nFeatures features
        train_x = all_train_x[selectedFeatures]
        test_x = all_test_x[selectedFeatures]
        
        #Run PCA
        pca = PCA(n_components = nFeatures);
        PCA_train_x = pca.fit_transform(train_x)
        PCA_test_x = pca.transform(test_x)
        
        #classifier initialization, training and testing
        clf = classifier
        clf.fit(PCA_train_x, train_y);
        score = clf.score(PCA_test_x, test_y);
        
        
        PCA_report[frozenset(selectedFeatures)] = score;
#        print "\nfeature SET: ", len(selectedFeatures)
#        print "score: ", score
    
    return PCA_report   



def get_PCA_report(filepath, classifier, nTr = 3, nTest = 10):
    """Get the report of featureSet size vs. %accuracy using Random Forest as the feature selection filter.
    PCA is applied after feature selection"""
    #1. Get standardized train and test data
    all_train_x, train_y, all_test_x, test_y = makeStdDataSets2(filepath, nTr, nTest);
    
    #Total number of features is the number of columns in train_x ( this should equal that of test_x)
    _, total = all_train_x.shape
#    2. select features with varying number of features
    PCA_report = {};
    for nFeatures in range(1, total +1):
        #Run PCA
        pca = PCA(n_components = nFeatures);
        reduced_train_x = pca.fit_transform(all_train_x)
        reduced_test_x = pca.transform(all_test_x)
        
        #classifier initialization, training and testing
        clf = classifier
        clf.fit(reduced_train_x, train_y);
        score = clf.score(reduced_test_x, test_y);
        
        
        PCA_report[nFeatures] = score;
#        print "\nfeature SET: ", nFeatures
#        print "score: ", score
    
    return PCA_report   
    
 
def show3D(filepath):
    """
    Given the per-user data with all features, we do the following:
    1. select the best three features using RF as a filter
    2. restrict the train-x and test-x to have only the selected top-three features
    3 (For the second graph only). Run PCA to rotate/transform the axes in 3-dimentional space
    4. Scatter plot the resulting points. Set the color index follow the label (class) of the point
    """
    train_x, train_y, test_x, test_y = makeStdDataSets2(filepath);
    selected = selectFeatureSet_RF(train_x, train_y, 3);
    train_x = train_x[selected]; test_x = test_x[selected];
    
    #Each feature column
    f0, f1, f2 = train_x[train_x.columns[0]], train_x[ train_x.columns[1]], train_x[ train_x.columns[2]]
    
    ##3d for raw with three best features
    fig1 = plt.figure(1, figsize = (8,6));
    ax1 = Axes3D(fig1)
    ax1.scatter(f0, f1, f2, c = train_y, cmap = plt.cm.Paired)
    ax1.set_xlabel (train_x.columns[0]); ax1.set_ylabel (train_x.columns[1]); ax1.set_zlabel(train_x.columns[2]);
    ax1.set_title('raw data plotted in 3D')
    
    
    #pca to choose three principle components and then plot on 3d
    PCA_train_x = PCA(n_components = 3).fit_transform(train_x)
    #Note: PCA.fit_trainsform returns a numpy array, not a dataframe.
    #    : The feature names are meaningless now ( we don't have them anyways).
    pc0 = PCA_train_x[:, 0]; pc1 = PCA_train_x[:, 1]; pc2 = PCA_train_x[:, 2];
    
    fig2 = plt.figure(2, figsize = (8,6));
    ax2 = Axes3D(fig2);
    ax2.scatter(pc0, pc1, pc2, c = train_y, cmap = plt.cm.Paired);
    ax2.set_title("First three Principle components");
    ax2.set_xlabel('first pc'); ax2.set_ylabel("second pc"); ax2.set_zlabel("third pc")
    plt.show()   
    
    
def plotTwoReports(title, report1, report2, toSave = True):    
    
    plt.figure()
    x1 = []; y1 = [];
    for k in report1.iterkeys():
        x1.append(len(k))
        y1.append(report1[k])
        
    x2=[]; y2= [];
    for j in report2.iterkeys():
        x2.append(len(j))
        y2.append(report2[j])
    
    plt.scatter(x1,y1, marker = 'x', c = 'b', s = 7, label='no pca');
    #plt.hold
    plt.scatter(x2,y2, marker = '+', c = 'r', s = 9, label='with pca');
    
    plt.title(title);
    plt.xlabel('number of selected features');
    plt.ylabel('%accuracy');
#    plt.xlim([0,24]);
#    plt.ylim([0,1.0]);
    
    axes = plt.gca()
    axes.set_xlim([0,26])
    axes.set_ylim([0,1.0])    
    
    plt.xticks(np.arange(0, 26, 1.0));
    plt.yticks(np.arange(0, 1.0, 0.1));    
    
    plt.legend(loc = 'best')
    
    if toSave:
        outDir = '..\\pca_effect\\' + clfName+ '\\'
        outName = fname +'.png'
        outPath = os.path.join(outDir, outName)
    
         #check if the outpath is already created.
        try:
            os.makedirs(outDir);
        except OSError:
            if not os.path.isdir(outDir):
                raise
        plt.savefig(outPath)
    plt.show()

    return
    
def plotThreeReports(title, report1, report2, report3, toSave = True):
    """Assume key of report1 and report2 are frozenset of selectedFeatureSet 
    (i.e. a list of selected column names), and key of report3 is nFeatures 
    (i.e. integer)
    """            
    plt.figure(figsize = (15,10))
    x1 = []; y1 = [];
    for k in report1.iterkeys():
        x1.append(len(k))
        y1.append(report1[k])
        
    x2=[]; y2= [];
    for j in report2.iterkeys():
        x2.append(len(j))
        y2.append(report2[j])
        
        
    x3=[]; y3= [];   
    for i,v in report3.iteritems():
        x3.append(i)
        y3.append(v)
    
    plt.scatter(x1,y1, marker = 'x', c = 'b', s = 11, label='fs_RF');
    plt.scatter(x2,y2, marker = '+', c = 'r', s = 11, label='fs_RF + pca');
    plt.scatter(x3,y3, marker = '.', c = 'k', s = 11, label='fs_pca');

    
    plt.title(title);
    plt.xlabel('number of selected features');
    plt.ylabel('%accuracy');
#    plt.xlim([0,24]);
#    plt.ylim([0,1.0]);
    
    axes = plt.gca()
    axes.set_xlim([0,26])
    axes.set_ylim([0,1.0])    
    
    plt.xticks(np.arange(0, 26, 1.0));
    plt.yticks(np.arange(0, 1.0, 0.1));    
    
    plt.legend(loc = 'best')
    
    if toSave:
        outDir = '..\\fs_comparison\\' + clfName+ '\\'
        outName = fname +'.png'
        outPath = os.path.join(outDir, outName)
    
         #check if the outpath is already created.
        try:
            os.makedirs(outDir);
        except OSError:
            if not os.path.isdir(outDir):
                raise
        plt.savefig(outPath)
    plt.show()

    return    

 
 
#######TEST##########################################################################################
#dirPath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\';
#for i in range(0,6):
#    fname = 'features_' +str(i)
#    fmt = '.csv'
#    filepath = dirPath + fname + fmt
#    #classifier = LinearSVC()
#    #clfName = 'linearSVC';
#    classifier = KNeighborsClassifier(1)
#    clfName = 'knn_1'
#    FS_report = get_FS_report(filepath, classifier);
#    e = time.time()
#    print 'runtime: ', e - s
#    plot_FS_report(FS_report,clfName, fname)



############To test pca effects######################################################################
#dirPath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\';
#
#for i in range(0,6):
#    fname = 'features_' +str(i)
#    fmt = '.csv'
#    filepath = dirPath + fname + fmt
##    classifier = LinearSVC()
##    clfName = 'linearSVC';
#    classifier = KNeighborsClassifier(1)
#    clfName = 'knn_1'
#    
#    #Get two reports (one without pca, the other with pca)
#    fs_report = get_FS_report(filepath, classifier)
#    pca_fs_report = get_PCA_FS_report(filepath, classifier)
#    pca_report = get_PCA_report(filepath, classifier)
#    title = fname + '\n' + clfName
#    plotThreeReports(title,fs_report,pca_fs_report, pca_report)
#    
#    
