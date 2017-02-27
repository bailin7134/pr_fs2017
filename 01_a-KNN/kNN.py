#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 23:12:19 2017

@author: lbai
"""

import numpy as np
import timeit as tm

K = 1

# read from train.csv & test.csv
trainDataSet = np.genfromtxt('train.csv', delimiter=",")
testDataSet = np.genfromtxt('test.csv', delimiter=",")

testDataRow, testDataCol = testDataSet.shape
trainDataRow, trainDataCol = trainDataSet.shape

#initialization
correctCase = 0

# calcualte the sum of square
#for i in range(1): #testDataRow):
#testDataRow = 5
#trainDataRow = 5
#trainDataSet = trainDataSet[0:5,:]

#i=1


for i in range(testDataRow):
    start = tm.default_timer()
    
    testEleVector = testDataSet[i,:]
    testEleMatrix = np.tile(testEleVector,trainDataRow).reshape(trainDataRow, trainDataCol)
    # euclidean distance
    diffSquareMatrix = np.square(testEleMatrix-trainDataSet)
    sumSquareVector = diffSquareMatrix[:,1:trainDataCol].sum(axis=1)

    # sort by index
    sumSquareVectorIndex = np.argsort(sumSquareVector)
    voteTable = np.array([0,0,0,0,0,0,0,0,0,0])
    # majority vote
    for j in range(K):
        index = sumSquareVectorIndex[j]
        voteNum = trainDataSet[index,0].astype(int)
        voteTable[voteNum] = voteTable[voteNum] + 1
    # make a decision
    predictNum = np.argmax(voteTable)
        
    # check the result
    if predictNum == testDataSet[i,:][0]:#.astype(int):
        correctCase = correctCase + 1
        print("Correct!")
        print(i)
    else:
        print("Wrong!")
        print(i)

    stop = tm.default_timer()
    print stop - start

# check the correct ratio
print("among all the test cases, the correct ratio is")
print np.divide(float(correctCase),float(testDataRow))
