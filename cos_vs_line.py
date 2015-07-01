# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:31:15 2015

@author: LLP-admin
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,np.pi,num= 20)
def f1(x):
    return -2*x/np.pi +1;
def f2(x):
    return np.cos(x);
fig = plt.figure(figsize = (6,8))
ax1 = fig.add_subplot(2,1,1);
ax1.set_xlabel('angle');

ax1.plot(x,f1(x), label = 'linear');
ax1.plot(x, f2(x), label = 'cosine');
ax1.plot(x, np.abs(f2(x) - f1(x)),color = "red", label = '|cos - lin|');

ax1.set_title("cosine vs linear")
ax1.legend(loc = "best")
ax1.plot(x, [0 for x_i in x], 'k--')
ax1.set_xlim([x[0], x[-1]])
ax1.set_ylim([-1,1])

#ax2 = fig.add_subplot(2,1,2)
#ax2.set_title("difference btw cosine and linear")

