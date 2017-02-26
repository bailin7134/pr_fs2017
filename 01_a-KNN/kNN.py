#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 23:12:19 2017

@author: lbai
"""

import numpy as np

# read from train.csv & test.csv
trainDataSet = np.genfromtxt('train.csv', delimiter=",")
testDataSet = np.genfromtxt('test.csv', delimiter=",")

testDataRow, testDataCol = testDataSet.shape

for i in range(testDataRow):
    np.tile(testDataSet[:,i],testDataCol)