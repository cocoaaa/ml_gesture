# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:09:50 2015

@author: LLP-admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:51:57 2015

@author: LLP-admin
"""
import os
from simple import *
from sklearn import svm
#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLogisticRegression;
from sklearn.linear_model import LogisticRegression

#feature selection
from sklearn.feature_selection import SelectKBest, f_regression

#scoring metric
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

#encoding
from sklearn.preprocessing import LabelEncoder

#plotting
import matplotlib.pyplot as plt
import filters 
        
def makeTestData1(filepath, maxTrainSess, nTestPerClass = 10):
    """
    Given the filepath to the per-user data set, make a test data set to have 
        the last 'x' number of instances from each class.
    Returns the dataFrame of the test data.
    This is to be used with getReport1
    """
    dict_df_class = divideByClass(filepath);
    dict_trainPerClass = {}
    test_set = pd.DataFrame();
    
    for (className, df_class) in dict_df_class.iteritems():
        length = len(df_class);        
        if (maxTrainSess  < length - nTestPerClass):
            print "WARNING: maxTrainSess can be too big for later training sessions."
            print "It may result in overlapping data instances in training and testing."
        upperbd = min(maxTrainSess, length-nTestPerClass-1)-1; #+1?
        
        #Add to the new dictionary for train set per class (value), with key = className.
        dict_trainPerClass[className] = df_class.iloc[range(0,upperbd)];
#        print "printing test df_class:\n ", df_class.iloc[range(length-numPerClass, length)]
#        print "updated to trian dic!"
        
        #Create the fixed sized, global test_set.
        test_set = test_set.append(df_class.iloc[range(length-nTestPerClass, length)], ignore_index = True);
#        print "Finished appending to the test_set";
#        print "Now the size of the test_set is: \n", test_set;
        
    return dict_trainPerClass, test_set;
  


def getReport1(filepath, classifier, maxTrainSess, nTestPerClass = 10):
    """
    Given the filepath, 
    """
    
    #Make the trainPerClass dictionary and the global test set.
    dict_trainPerClass, test_set = makeTestData1(filepath, maxTrainSess, nTestPerClass);
    sampleDF = dict_trainPerClass.values()[0];
    columns = sampleDF.columns;
#    assert (test_set.columns == columns);
    
    #Initialize the classifier with parameters
    classifier = classifier; #ToDo: parameter setting. getOtions?
    
    #Initialize the record.  
    #Key = one-based index of the last run, Value = %accuracy of the classifier after the last training.
    report = {};
    #Prepare the test set.
    rawTest_x, rawTest_y = splitXY(test_set);
    
    #nInstances: we will choose the minimum
    #print "list:" ,[len(trainPerClass) for trainPerClass in dict_trainPerClass.itervalues()]
    
    n_tr= min([len(trainPerClass) for trainPerClass in dict_trainPerClass.itervalues()]);
#    print "max number of training sessions possible is: ", n_tr;
    uppbd = min(n_tr, maxTrainSess)
    
    for j in range(1, uppbd + 1):
        #Initialize the batch for training.
        batch =  pd.DataFrame(columns = columns);

        for trainPerClass in dict_trainPerClass.itervalues():
            batch = batch.append(trainPerClass.iloc[0:j]);
        
        #split the batch into features and labels
        batch_x, batch_y = splitXY(batch);
        
        #Standardize the train data.  Apply the mean and std parameter to scale the test data accordingly.
        std_batch_x, std_test_x = piped_standardize(batch_x, rawTest_x);
        
        #ToDo: Apply filters here
        #Filter1: remove constant feature colms
#        selectedCols = filters.notConstantCols(std_batch_x);
        isNotConst = np.array((np.std(std_batch_x) != 0 ))
        
        #Combine the filters
        isSelectedCol = isNotConst#isRFE & isNotConst;
#        print "size match?: ", len(isSelectedCol) == len(std_batch_x.columns)     
        
        
        ######Don't need to modify even after new filters#################
        #selectedCols is a list of column names that are selected.
        selectedCols = [std_batch_x.columns[i] for (i,v) in enumerate(isSelectedCol) if v == 1]
#        print 'Selected Cols: ', selectedCols
        
        filtered_batch_x = std_batch_x[selectedCols];
        filtered_test_x = std_test_x[selectedCols];

        
        #For now
        train_x = filtered_batch_x; train_y = batch_y;
        test_x = filtered_test_x; test_y = rawTest_y;
        
        #train_y and test_y's index must be in order 
        train_y.index = range(0, len(train_y));
        test_y.index = range(0, len(test_y));
        assert(len(train_x.columns) == len(test_x.columns));
        
        
        #Don't need to modify even after more filtering is applied later
        #train the classifier on this batch
        classifier.fit(train_x, train_y);
    
        #test the classifier on the fixed test set
        score = classifier.score(test_x, test_y);
        
        #record the accuracy (%)
        report[j] = score;
#        print 'report', report 

    return report

def selectFeatureSet_anova(data_x, data_y, nFeatures):
    """
    Use cross-validation with nfolds < nsamples in test_x (i.e. nTestPerClass (defualt 10) * nClasses (eg 12))
    Select best features based on ANOVA for svm.
    """
    #1. Run SVM to get the feature ranking
    anova_filter = SelectKBest(f_regression, k= nFeatures)
    anova_filter.fit(data_x, data_y)
    print 'selected features in boolean: \n', anova_filter.get_support()
    print 'selected features in name: \n', test_x.columns[anova_filter.get_support()];
    
    #2. Select the top nFeatures features
    selectedCols = data_x.columns[anova_filter.get_support()]
    #3. Run SVM (or any other) again on this selected features
    return selectedCols
    
def selectFeatureSet_RF(data_x, data_y, nFeatures):
    rf_filter = RandomForestClassifier(max_features = 'auto')
    rf_filter.fit(data_x, data_y);
    rankings = rf_filter.feature_importances_;
    selectedBool = np.argsort(rankings)[-nFeatures:]
#    selectedBool = sorted(range(len(rankings)), key = lambda x: rankings[x])[-nFeatures:];
    return data_x.columns[selectedBool]
 
        
def evalThisFS(train_x, train_y, test_x, test_y, classifier, selectedCols):
        
    if len(selectedCols) == 0:
        score = 0.0
        
    train_x = train_x[selectedCols];
    test_x = test_x[selectedCols];
    
    
    #Don't need to modify even after more filtering is applied later
    #train the classifier on this batch
    classifier.fit(train_x, train_y);

    #test the classifier on the fixed test set
    score = classifier.score(test_x, test_y);
    return (selectedCols, score);
        
        
def getReport2(filepath, clf, maxTrainSess, nTestPerClass = 10, showReport = True):
    dict_dataPerClass = divideByClass(filepath);
    sampleDF = dict_dataPerClass.values()[0];
    columns = sampleDF.columns
#    np_dataPerClass = [np.array(dataPerClass) for dataPerClass in dict_dataPerClass.values()]
    no_filter_report= {};  
    yes_filter_report = {};      
    #Initialize the batch for training.
      
    for j in range(1, maxTrainSess +1):
        #Clear out the test_set for each training session
        test_set = pd.DataFrame(columns = columns);
        batch =  pd.DataFrame(columns = columns)

        for dataPerClass in dict_dataPerClass.itervalues():
#            assert( not(dataPerClass.isnull().any().any())  ) ; print 'No None in this class dataset!'
            batch = batch.append(dataPerClass.iloc[0:j]);
            #Now, need to prepare the test data set.
            test_set = test_set.append( dataPerClass.iloc[j+1:j+nTestPerClass+1] )
            
        
        #split the batch into features and labels
        batch_x, batch_y = splitXY(batch)
        rawTest_x, rawTest_y = splitXY(test_set)
        #Done creating training and test data sets for this session.
            
        #Standardize the train data.  Apply the mean and std parameter to scale the test data accordingly.
        std_batch_x, std_test_x = piped_standardize(batch_x, rawTest_x);
        
        #Label encoding
#        batch_y.index = range(0, len(batch_y))
        #ToDo: Efficiency-wise, we should do this only once when the entire dataset is first
        #imported.  
        le = LabelEncoder()
        le.fit(batch_y);
        
        encBatch_y = le.transform(batch_y)
        encTest_y = le.transform(rawTest_y) 
        
    
        #ToDo: Apply filters here:
        #ToDo: do feature selections independent of the learning method.
        #Filter1: remove constant feature colms
#        selectedCols = filters.notConstantCols(std_batch_x);
        isNotConst = np.array((np.std(std_batch_x) != 0 ))
        
        #Combine the filters
        isSelectedCol = isNotConst

        ######Don't need to modify even after new filters#################
        #selectedCols is a list of column names that are selected.
    
        #selectedCols is the list of strings (i.e. selected column names)    
        selectedCols = [std_batch_x.columns[i] for (i,v) in enumerate(isSelectedCol) if v == 1]
#        print 'Selected Cols: ', selectedCols
        
        filtered_batch_x = std_batch_x[selectedCols];
        filtered_test_x = std_test_x[selectedCols];

        
        #For now
        train_x = filtered_batch_x; train_y = encBatch_y
        test_x = filtered_test_x; test_y = encTest_y
        
        #Make sure there is no NaN in train_x, train_y, test_x, test_y
#        assert( not(train_x.isnull().any().any()) ); print 'train-x pass!';
#        assert( not(train_y.isnull().any().any()) ) ; print 'train-y pass!'
#        assert( not(test_x.isnull().any().any())  ) ; print 'test-x pass!'
#        assert( not(test_y.isnull().any().any())  ) ; print 'test-y pass!'

        
        #Make sure the number of features in train_x and test_x are same
        assert(len(train_x.columns) == len(test_x.columns));
        
        
        #Don't need to modify even after more filtering is applied later
        #train the classifier on this batch
        classifier = clf
        classifier.fit(train_x, train_y);
#        print 'selected features: ', classifier.feature_importances_
    
        #test the classifier on the fixed test set
        score = classifier.score(test_x, test_y);
        #record the accuracy (%)
        no_filter_report[j] = score;
        print 'nofilter score: ', score
        if showReport:
            pred_y = classifier.predict(test_x);
            print '\n'+classification_report(test_y, pred_y)
#            print 'f_1 score avgerage, macro: ', f1_score(test_y, pred_y, average='macro')
    
        
        ###########DO SVM or RandomForest Filtering######################################
        ###################################################################
    
        ###################################################################
        selectedCols = selectFeatureSet_RF(train_x, train_y, nFeatures = 10)
        newClassifier = clf#Reset back to initial (blank) classifier
        yes_filter_report[j] = evalThisFS(train_x, train_y,test_x, test_y, newClassifier, selectedCols )
        print 'filter score :', yes_filter_report[j]
#        print 'report', report 
        
        
    return no_filter_report,yes_filter_report


def perUserEvaluations(dirPath, classifier, classifierName, maxTrainSess, nTestPerClass = 10):
    """
    Inputs:
    1. dirPath: String path to the directory    
    2. classifier: sklearn classifier object
    3. classifierName: String of the classifier's name
    4. nTestPerClass: number of tests per class (total test set size is 10*number of class values). 
                    (Default is fixed to ten)
                    
    Outputs:                
    1. Outputs graphs of the report, one graph per dataset in the directory.
    2. Returns a list of peak %accuracy for each per-user data set.
    """
           
    
############Inside helper function#########################
###########################################################    
    def showSaveReport(report, toSave = True):
        """Given the report dictionary,
        plot the graph with x-aix: number of training batch, y-axis %accuracy.
        """    
#        print classifier.name??
        plt.figure()
        plt.plot(report.keys(), report.values());
        
        plt.title("Report on: "+ fname+ \
                "\nClassifier: "+ clfName +\
               "\nFirst Peak: " + str(p_accuracy) + " at " + str(p_idx) );
        plt.xlim([0,17]);plt.xticks(np.arange(0, 17, 1.0));
        plt.ylim([0,1.0]);plt.yticks(np.arange(0, 1.0, 0.1));
        plt.xlabel("number of training batch")
        plt.ylabel("% accuracy")
        
        #If you want to save the graphs
        if toSave:
            #Set up output path
            outDir = '..\\perUser_evaluations\\' + clfName+ '\\'
            outName = fname +'.png'
            outPath = path.join(outDir, outName)
            
            #check if the outpath is already created.
            try:
                os.makedirs(outDir);
            except OSError:
                if not os.path.isdir(outDir):
                    raise
            
            #Save and show the result
            plt.savefig(outPath) #classifierName
        plt.show();
        plt.close();
        return
########################################################
    clfName = classifierName;
    #Initializa peaks: a list of peak info for each per-user data set.    
    peaks = [];
    
    #Run evaluation and graph report for each per-user data in the directory.     
    for i in range(0,6):
        fname = "user_" + str(i) 
        filepath = dirPath + fname + ".csv";
            
        #Get the report for the data set
        report = getReport2(filepath, classifier, maxTrainSess, nTestPerClass = 10, showReport = False);
        
        #Get peak information.
        (p_idx, p_accuracy) = findFirstPeak(report);
        peaks.append(p_accuracy);
        #Graph the report
        showSaveReport(report);
    
    return peaks
    
#########################################################################################
##################################################################################
###################################################################################
###Test for runEvaluation###
##Initialize the classifier
##clf = KNeighborsClassifier(1);
#clf = svm.LinearSVC()
##clf = LogisticRegression
#clfName = 'linearSVC'
#clfName = 'RandomForest'
###classifier =DecisionTreeClassifier();
#classifier = RandomForestClassifier();
##################################to run
#dirPath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\';
#fname = 'user_' + str(0)
#fmt = '.csv'
#filepath = dirPath + fname + fmt
#peaks = perUserEvaluations(dirPath, clf, clfName, maxTrainSess = 17);
#print "peak lists: ", peaks
##############################################
#for i in range(0,6):
#    fname = 'features_' + str(i)
#    fmt = '.csv'
#    filepath = dirPath + fname + fmt
#    report1 =  getReport1(filepath, clf, maxTrainSess = 17, nTestPerClass = 10);
#    report2 =  getReport2(filepath, clf, maxTrainSess = 17, nTestPerClass = 10);
#    plt.figure(0);
#    plt.plot(report1.keys(), report1.values(), label='fixed testset' )
#    plt.hold;
#    plt.plot(report2.keys(), report2.values(), label = 'subseq. testset');
#    
#    plt.title(fname + '\n'+clfName)
#    plt.legend(loc='best')
#    length = max(len(report1), len(report2))
#    plt.xlim([1,length]);plt.xticks(np.arange(1, length, 1.0));
#    
#    plt.ylim([0,1.0]);plt.yticks(np.arange(0, 1.0, 0.1));
#    
#    
#    #Set up output path
#    outDir = '..\\compareTestTypes\\' + '\\' + clfName + '\\'
#    outName = fname+'_'+ clfName + '.png'
#    outPath = outDir + outName
#    
#    #Save and show the result
#    plt.savefig(outPath) #classifierName
#    plt.show();
#    plt.close()
#    
#    
#To test the 'showReport' edit to getReport2
#The printed report gives info about the classes (how correctly classified each of the members are).
#dirPath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\';
#fname = 'user_' + str(0)
#fmt = '.csv'
#filepath = dirPath + fname + fmt
#clf = svm.LinearSVC()
#clfName = 'linearSVC'
#
#report, filterReport = getReport2(filepath, clf, maxTrainSess = 3, nTestPerClass = 10, showReport = True)    
#print report.values()
#print [v[1] for v in filterReport.values()]
#    
#
