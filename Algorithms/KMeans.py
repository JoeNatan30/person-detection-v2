#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:27:15 2018

@author: joe
"""

import numpy as np
from sklearn.cluster import KMeans

def iniciar(pc):
    
    pc_arr = pc.to_array()
    
    X = np.array(pc_arr)
    
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    return kmeans.labels_

'''  
array = np.array([[1,1,1],
                  [3,3,3],
                  [1,1,2],
                  [3,2,3]])
''' 