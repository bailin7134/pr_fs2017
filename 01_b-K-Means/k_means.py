#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:25:57 2017

@author: Lin Bai, 09935404
"""

import numpy as np

# read from train.csv & test.csv
trainDataSet = np.genfromtxt('train.csv', delimiter=",")
testDataSet = np.genfromtxt('test.csv', delimiter=",")

K = 5

def function_k_means(K, trainDataSet):

    testDataRow, testDataCol = testDataSet.shape
    trainDataRow, trainDataCol = trainDataSet.shape

    # initialize K centroids
    centroids = np.random.rand(K, trainDataCol-1)*255
    # initialize centroid storage array
    centroids_belong = np.zeros(trainDataRow)
    
    # cost function
    cost_function = np.empty(trainDataCol-1)
    cost_function.fill(255)
    cost_function = np.sum(np.square(cost_function))*K
    cost_function_new = 0

    # if cost function stays
    if (cost_function_new >= cost_function):

        for i in range(1, trainDataRow):
            # calculate the distance to each centroid
            distance = np.dot(centroids, trainDataSet[i, 1:trainDataCol])
            # find the nearest centroid
            min_index = np.argmin(distance)
            centroids_belong[i] = min_index

        # update the centroid position
        cost_function_new = 0
        for j in range(K):
            sum_position = np.zeros(trainDataCol-1)
            sum_number = 0
            for element in range(trainDataRow):
                if centroids_belong[element] == j:
                    sum_position = sum_position + trainDataSet[element, 1:trainDataCol]
                    sum_number = sum_number + 1

            centroids[j] = sum_position/sum_number

            # calculate new cost function
            for element in range(trainDataRow):
                if centroids_belong[element] == j:
                    cost_function_new = cost_function_new + np.sum(np.square(trainDataSet[element, 1:trainDataCol]-centroids[j]))
