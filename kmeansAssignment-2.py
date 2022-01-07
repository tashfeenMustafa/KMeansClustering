# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 02:17:04 2021

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 01:19:06 2021

@author: LENOVO
"""

import random
import math
import csv
#import json
#import sys
import time
import numpy as np

def load(file_name):
    # data(list of list): [[index, dimensions], [.., ..], ...]
    data = []              
    fh = open(file_name)
    for line in fh:
        line = line.strip().split(',')
        temp = [int(line[0])]
        for feature in line[1:]:
            temp.append(float(feature))
        data.append(temp)  
    return data

def get_sample(data):
    length = len(data)
    sample_size = int(length * 0.01)
    random_nums = set()
    sample_data = []
    for i in range(sample_size):
        random_index = random.randint(0, length - 1)
        while random_index in random_nums:
            random_index = random.randint(0, length - 1)
        random_nums.add(random_index)
        sample_data.append(data[random_index])
    return sample_data

def initialize_centroids(data, dimension, k):
    centroids = [[0 for _ in range(dimension)] for _ in range(k)]
    max_feature_vals = [0 for _ in range(dimension)]
    min_feature_vals = [0 for _ in range(dimension)]

    # TO DO
    # Calculate max feature and min feture value for each dimension
    for i in range(dimension):
        for j in range(len(data)):
            if data[j][i] > max_feature_vals[i]:
                max_feature_vals[i] = data[j][i]
            if data[j][i] < min_feature_vals[i]:
                min_feature_vals[i] = data[j][i]
    
    #diff: max - min for each dimension
    diff = []
    for i in range(len(max_feature_vals)):
        diff.append(max_feature_vals[i] - min_feature_vals[i])

    # for each centroid, in each dimension assign centroids[j][i] = min_feature_val + diff * random.uniform(1e-5, 1)   
    for i in range(len(centroids)):
        for j in range(len(centroids[i])):
            centroids[i][j] = min_feature_vals[j] + diff[j] * random.uniform(1e-5, 1)
    
    return centroids

def get_euclidean_distance(p1, p2):
    distance = 0
    s = []
    #Write your code
    for i in range(len(p1)):
        s.append((p1[i] - p2[i]) ** 2)
    sum_of_s = 0
    for i, num in enumerate(s):
        sum_of_s += num
    distance = math.sqrt(sum_of_s)
    return distance

def get_new_centroid(centroid, point, cluster_points):
    point = list(point)
    new_centroid = []
    for i in range(len(centroid)):
        new_centroid.append((centroid[i] + point[i]))
    
    for i in range(len(new_centroid)):
        new_centroid[i] = new_centroid[i] / cluster_points
    
    return new_centroid

def kmeans(data, dimension, k):
    #centroids: [(centroid0 fearures); (centroid1 features); ... ..]
    centroids = initialize_centroids(data, dimension, k)

    #cluster_affiliation: [((point1index  features),clusterindex); ((point2index features), clusterindex)... ]
    cluster_affiliation = [[tuple(features), None] for features in data]
    count = 1
    
    flag = 1
    jPrev = None
    
    while flag:
        print('count: ', count)
        for i, point in enumerate(data):
            #initializing min_distance and min_distance_index
            min_distance = float('inf')
            min_distance_index = None
            
            #find closest centroids for each data points
            for cluster_index, centroid in enumerate(centroids):
                if centroid[0] == None:
                    continue
                distance = get_euclidean_distance(centroid, point[1:])
                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = cluster_index
            #record or update cluster for each data points
            if cluster_affiliation[i][1] != min_distance_index:
               cluster_affiliation[i][1] = min_distance_index        
        
        #recompute centroid
        centroids = [[0 for _ in range(dimension)] for _ in range(k)]
        cluster_point_count = [0 for _ in range(k)]
        
        #TO DO
        #write your code to count each cluster pointcount and store them in clutser_point_count structure 
        # counting cluster_point_count
        for i in range(k):
            for affiliation in cluster_affiliation:
                if affiliation[1] == i:
                    cluster_point_count[i] += 1                  
        print(cluster_point_count)
          
        #recompute new centroids using the count
        for affiliation in cluster_affiliation:
            for cluster_point_index, cluster_point in enumerate(cluster_point_count):
                if affiliation[1] == cluster_point_index:
                    centroids[cluster_point_index] = get_new_centroid(centroids[cluster_point_index], affiliation[0], cluster_point_count[cluster_point_index])  
        print(centroids)
        
        #TO DO 
        #Terminate the while loop based on termination criteria. Write your code to turn flag = false
        sum_of_all_min = 0
        for index, point in enumerate(cluster_affiliation):
            val = np.subtract(list(point[0][1:]), centroids[point[1]])
            abs_val =  [np.linalg.norm(elem) ** 2 for elem in val]
            sum_of_all_min += min(abs_val)
            
        J = (1/dimension) * sum_of_all_min
        print(J)
        if jPrev:
            if abs(J - jPrev) <= J * (10 ** -5):
                flag = False
        else:
            jPrev = J
        
        count += 1
    
    return (centroids, cluster_affiliation)

def write_centroids(centroids, output_file):
    file = open(output_file, 'w+', newline = '') 
    with file:     
        write = csv.writer(file) 
        write.writerows(centroids) 

def write_cluster_affiliation(cluster_affiliation, output_file):
    file = open(output_file, 'w+', newline = '')
    writer = csv.writer(file)
    
    for i in range(len(cluster_affiliation)):
        writer.writerow([cluster_affiliation[i][0][0], cluster_affiliation[i][1]])

def main():
    start = time.time()    
    #input path
    input_path = 'E:\IUB-Courses\Autumn 2021\Data Mining & Warehouse\Assignment 2\K-means Clustering\data'    
    K = 4 # K clusters
    output1 = 'out1-ic.csv' 
    output2 = 'out2-ic.csv'     
    #getting data by loading data from input path file
    data_num = 0
    data = load(input_path + '/data' + str(data_num) + '.txt')    
    dimension = len(data[0]) - 1
    
    #sampling data from the data file
    sample_data = get_sample(data)
    
    #centroids: [(centroid1 features); (centroid2 features)]
    #cluster_affiliation: [((point1 features with point index),group index); ((point2 features with point index),group index)... ]
    	
    centroids, cluster_affiliation = kmeans(sample_data, dimension, K)
    	
    # TODO   
    #Write all final centroids in out1 file. one line for each centroids, Features would separted by ,
    write_centroids(centroids, output1)
    #in out2, Write the cluster assignments for each of the point. Each line: Point index, cluster number
    write_cluster_affiliation(cluster_affiliation, output2)    	
    
    print('Duration: %s' % (time.time() - start))

if __name__ == "__main__":
	main()
			
