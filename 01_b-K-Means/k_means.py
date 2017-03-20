#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:27:42 2017

@author: lbai
"""

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

def function_k_means(K, trainDataSet):
    
    # testDataRow, testDataCol = testDataSet.shape
    trainDataRow, trainDataCol = trainDataSet.shape
    
        
    # initialize K centroids
    centroids = np.random.rand(K, trainDataCol-1)*255
    # initialize centroid storage array
    centroids_belong = np.zeros(trainDataRow)
        
    # cost function
    cost_function = np.square(255)*trainDataRow*(trainDataCol-1)
    cost_function_new = cost_function - 1
    iteration = 0
    
    # if cost function stays
    while (cost_function_new < cost_function):
    
        for i in range(trainDataRow):
            # calculate the distance to each centroid
            distance = np.sum(np.power(centroids - trainDataSet[i, 1:trainDataCol], 2),1)
            # find the nearest centroid
            min_index = np.argmin(distance)
            centroids_belong[i] = min_index
                
        centroids_belong = centroids_belong.astype(int)
        
        # update the centroid position
        sum_position = np.zeros((K, trainDataCol-1))
        sum_number = np.zeros(K)
        for element in range(trainDataRow):
            sum_position[centroids_belong[element]] = sum_position[centroids_belong[element]] + trainDataSet[element, 1:trainDataCol]
            sum_number[centroids_belong[element]] = sum_number[centroids_belong[element]] + 1
        
        for j in range (K):
            if sum_number[j] != 0:
                centroids[j] = sum_position[j]/sum_number[j]
                
        # calculate new cost function
        cost_function = cost_function_new
        cost_function_new = 0
        for element in range(trainDataRow):
            cost_function_new = cost_function_new + np.sum(np.square(trainDataSet[element, 1:trainDataCol]-centroids[centroids_belong[element]]))
        
        iteration = iteration + 1
        #print iteration
        #print cost_function
        #print cost_function_new
        return centroids, centroids_belong


def dunn_index(K, trainDataSet, centroids, centroids_belong):
    
    trainDataRow, trainDataCol = trainDataSet.shape
    
    element_labels = [[] for y in range(K)]
    # category according to index
    for element in range(trainDataRow):
        element_labels[centroids_belong[element]].append(element)

    min_distances = 255*np.sqrt(trainDataCol-1)
    # Calculates the distances between the two nearest points of each cluster
    for i in range(K):
        for j in range(i+1, K):
            if ((not element_labels[i]) and (not element_labels[j])):
                continue
            else:
                for p in range(len(element_labels[i])):
                    element_p = trainDataSet[element_labels[i][p], 1:trainDataCol]
                    for q in range(len(element_labels[j])):
                        element_q = trainDataSet[element_labels[j][q], 1:trainDataCol]
                        dist_new = np.sqrt(np.sum(np.power(element_p-element_q,2)))
                        if dist_new < min_distances:
                            min_distances = dist_new
                        
    # Calculates cluster diameters (the distance between the two farthest data points in a cluster)
    max_distances = [[0] for y in range(K)]
    for i in range(K):
        if not element_labels[i]:
            continue
        else:
            for p in range(len(element_labels[i])):
                element_p = trainDataSet[element_labels[i][p], 1:trainDataCol]
                for q in range(p+1, len(element_labels[i])):
                    element_q = trainDataSet[element_labels[i][q], 1:trainDataCol]
                    dist_new = np.sqrt(np.sum(np.power(element_p-element_q,2)))
                    if dist_new > max_distances[i]:
                        max_distances[i] = dist_new
    
    delta_max = np.max(max_distances)
    
    D = min_distances/delta_max
    return D

def david_bouldin_index(K, trainDataSet, centroids, centroids_belong):
    
    trainDataRow, trainDataCol = trainDataSet.shape
    
    element_labels = [[] for y in range(K)]
    # category according to index
    for element in range(trainDataRow):
        element_labels[centroids_belong[element]].append(element)
    
    num = np.zeros(K)
    dist = np.zeros(K)
    # d_i calculation
    for i in range(K):
        center = centroids[i]
        for element in range(len(element_labels[i])):
            point = trainDataSet[element_labels[i][element], 1:trainDataCol]
            dist[i] = dist[i] + np.sum(np.power(point - center, 2))
            num[i] = num[i] + 1
    
    for i in range(K):
        if num[i] != 0:
            dist[i] = dist[i]/num[i]
    
    d_ij = [[np.inf for x in range(K)] for y in range(K)]
    
    # d(m_i, m_j)
    r_i_max = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                d_ij[i][j] = np.sum(np.power(centroids[i] - centroids[j], 2))
                # R_ij
                r_i = (dist[i] + dist[i])/d_ij[i][j]
                if r_i > r_i_max[i]:
                    # R_i
                    r_i_max[i] = r_i
                    
    DB = np.sum(r_i_max)/K
    return DB

K_list = [5, 7, 9, 10, 12, 15]
for K in K_list:
    print "When K = "
    print K
    [centroids, centroids_belong] = function_k_means(K, trainDataSet)
    print "The Dunn index is: "
    print dunn_index(K, trainDataSet, centroids, centroids_belong)
    print "The David Bouldin index is: "
    print david_bouldin_index(K, trainDataSet, centroids, centroids_belong)
    
