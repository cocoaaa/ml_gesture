# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:32:40 2015

@author: LLP-admin
"""

from itertools import chain, combinations

def g_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in xrange(len(s)+1))
l=[1,2,3,4]
g= g_powerset(l)
for i in range(0,10):
    print list(g.next())
