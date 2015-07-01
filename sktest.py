# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:41:49 2015

@author: LLP-admin
"""

from sklearn import svm, cross_validation, datasets
import pylab as pl

iris = datasets.load_iris()
X, y = iris.data, iris.target
model = svm.SVC()
cv_scores = cross_validation.cross_val_score(model, X, y, scoring='accuracy')
pl.plot(range(0,len(cv_scores)),cv_scores)
pl.ylim([0,1])
pl.show()