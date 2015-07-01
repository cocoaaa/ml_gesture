# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:14:02 2015
Confusion matrix and Class Reduction 
@author: LLP-admin
"""
from sklearn.metrics import confusion_matrix
import scripts\feature_selection 


def show_cm(train_x, train_y, test_x, test_y, classifier):
    y_pred = classifier.fit(train_x, train_y).predict(test_x)
    
    #Compute confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)

##############test show_cm##########################################
dirPath = 'C:\\Users\\LLP-admin\\workspace\\weka\\token-experiment-dataset\\';

for i in range(0,6):
    if i != 0:
        break
    fname = 'features_' +str(i)
    fmt = '.csv'
    filepath = dirPath + fname + fmt
#    classifier = LinearSVC()
#    clfName = 'linearSVC';
    classifier = KNeighborsClassifier(1)
    clfName = 'knn_1'
    
    #Get two reports (one without pca, the other with pca)
    raw_tr_x, tr_y, raw_test_x, test_y =makeStdDataSets2(filepath, nTr = 3, nTest = 10):
    fs_report = get_FS_report(filepath, classifier)
    pca_fs_report = get_PCA_FS_report(filepath, classifier)
    pca_report = get_PCA_report(filepath, classifier)
    title = fname + '\n' + clfName
    plotThreeReports(title,fs_report,pca_fs_report, pca_report)
    
    