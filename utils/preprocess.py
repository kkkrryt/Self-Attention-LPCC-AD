import os
from random import shuffle
import random
import numpy as np
import glob
import sys

from numpy.core.fromnumeric import resize
from NaiveOctree import bfs_process_octree
sys.path.append("../")
sys.path.append("../pybind/")
import time
# import pybind.myutils as myutils
from queue import Queue

def flattenFeatures(node):
    return (node.location[0],node.location[1],node.location[2],node.depth,node.curIndex,node.parentOccupancy)

def generateDataFromOctree_transformer_cpp(root, sequence_size, shuffle=True):

    start1 = time.time()
    results_cpp = bfs_process_octree(root)
    nodedict = results_cpp.nodedict
    layer_indexs = results_cpp.layerIndexs
    maxIndex = results_cpp.maxIndex
    maxLayer = results_cpp.maxLayer

    end1 = time.time()
    layer_indexs[maxLayer+1] = maxIndex # trick
    results = []
    labels = []
    for i in range(1,maxLayer):
        start = layer_indexs[i]
        end = layer_indexs[i+1] -1
        one_data = []
        one_label  = []
        for j in range(start,end+1):
            one_data.append(flattenFeatures(nodedict[j]))
            one_label.append(nodedict[j].occupancy)
            if(len(one_data) == sequence_size):
                results.append(one_data)
                labels.append(one_label)
                one_data = []
                one_label = []
        if(len(one_data) != 0):
            results.append(one_data)
            labels.append(one_label)
    
    if(shuffle):
        random.seed(123)
        randnum = random.randint(0,1000)
        random.seed(randnum)
        random.shuffle(results)
        random.seed(randnum)
        random.shuffle(labels)

    return (results,labels)

    
    

