#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:27:00 2019

@author: joe
"""

import numpy as np
from random import randint

"""
return a simple randint
"""
def randomIntBetween(mini,maxi):
    
    return randint(mini, maxi)

"""
creates a new array
assign value of the taken array
return the new array
"""
def assignArray3d(arr):
    
    newArr = []
    
    newArr.append(arr[0])
    newArr.append(arr[1])
    newArr.append(arr[2])
    
    return newArr

"""
creates a new array
assign these three values
return the new array
"""
def assignTo3dArray(value_1,value_2,value_3):
    
    newArr = []
    
    newArr.append(value_1)
    newArr.append(value_2)
    newArr.append(value_3)
    
    return newArr

"""
return negative of the value
"""
def negative(point): 
    return point * (-1)

def absolute(val):   
    return abs(val)

