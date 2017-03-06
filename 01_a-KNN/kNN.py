#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:46:48 2017

@author: Lin Bai, 09935404
"""

import numpy as np
#import timeit as tm

# read from train.csv & test.csv
trainDataSet = np.genfromtxt('train.csv', delimiter=",")
testDataSet = np.genfromtxt('test.csv', delimiter=",")


def fucntion_kNN(K, distance, trainDataSet, testDataSet):

    testDataRow, testDataCol = testDataSet.shape
    trainDataRow, trainDataCol = trainDataSet.shape

    #initialization
    correctCase = 0

    for i in range(testDataRow):
#        start = tm.default_timer()

        testEleVector = testDataSet[i,:]
        testEleMatrix = np.tile(testEleVector,trainDataRow).reshape(trainDataRow, trainDataCol)
        if distance == "euclidean":
            # euclidean distance
            diffSquareMatrix = np.square(testEleMatrix-trainDataSet)
            sumVector = diffSquareMatrix[:,1:trainDataCol].sum(axis=1)
        else:
            # manhhatan distance
            sumVector = np.sum(np.absolute(testEleMatrix-trainDataSet), axis=1)

        # sort by index
        sumVectorIndex = np.argsort(sumVector)
        voteTable = np.array([0,0,0,0,0,0,0,0,0,0])
        # majority vote
        for j in range(K):
            index = sumVectorIndex[j]
            voteNum = trainDataSet[index,0].astype(int)
            voteTable[voteNum] = voteTable[voteNum] + 1
        # make a decision
        predictNum = np.argmax(voteTable)

        # check the result
        if predictNum == testDataSet[i,:][0]:#.astype(int):
            correctCase = correctCase + 1
            # print("Correct!")
            # print(i)
        # else:
            # print("Wrong!")
            # print(i)

#        stop = tm.default_timer()
        # print stop - start

    # check the correct ratio
    print("K = ")
    print(K)
    print("distance = ")
    print(distance)
    print("among all the test cases, the correct ratio is")
    print np.divide(float(correctCase),float(testDataRow))


K_list = [1,3,5,10,15]
distance_list = ["euclidean", "manhhatan"]
for K in K_list:
    for distance in distance_list:
        fucntion_kNN(K, distance, trainDataSet, testDataSet)

