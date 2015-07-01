# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:12:44 2015
Confusion matrix studies on the Iris data
@author: LLP-admin
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets

df =datasets.load_iris()
data_x = pd.DataFrame( df.data, columns = df.feature_names )
f0 = data_x[ data_x.columns[[0]] ]; f1 = data_x [data_x.columns[1]]
f2 = data_x[ data_x.columns[2] ]
data_y = df.target

x_min = np.min(f0) - 0.5; x_max = np.max(f0) + 0.5;
y_min = np.min(f1) - 0.5; y_max = np.max(f1) + 0.5;

plt.figure(0,figsize = (8,6));
plt.clf()
x = np.arange(0,11,step = 0.1);
y = [el**2 for el in x]
plt.scatter(f0, f1, c = data_y, cmap = plt.cm.Paired)
plt.xlim = [x_min, x_max]; plt.xlabel('sepal length (cm)')
plt.ylim = [y_min, y_max]; plt.ylabel('sepal width (cm) ')

##3d
fig1 = plt.figure(1, figsize = (8,6));
ax1 = Axes3D(fig1)
ax1.scatter(f0, f1, f2, c = data_y, cmap = plt.cm.Paired)
ax1.set_xlabel (data_x.columns[0]); ax1.set_ylabel (data_x.columns[1]); ax1.set_zlabel(data_x.columns[2]);
ax1.set_title('raw data plotted in 3D')

##pca
PCA_data_x = PCA(n_components = 3).fit_transform(data_x)
pc0 = PCA_data_x[:, 0]; pc1 = PCA_data_x[:, 1]; pc2 = PCA_data_x[:, 2];

fig2 = plt.figure(2, figsize = (8,6));
ax2 = Axes3D(fig2);
ax2.scatter(pc0, pc1, pc2, c = data_y, cmap = plt.cm.Paired);
ax2.set_title("First three Principle components");
ax2.set_xlabel('first pc'); ax2.set_ylabel("second pc"); ax2.set_zlabel("third pc")
plt.show()



