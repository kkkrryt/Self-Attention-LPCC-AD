import os
from random import randint
import sys

from torch.nn.modules.activation import ReLU
sys.path.append("../")
import glob
import numpy as np
from utils.preprocess import generateDataFromOctree_transformer_cpp
import time
import pybind.myutils as myutils
import open3d as o3d

def readOctreeNpy(url):
    nowData = np.load(url,allow_pickle=True)

def generateOctreeNpy(sourceRoot, targetRoot, type, max_depth, sequence_size, shuffle=True):

    print("sourceRoot: {}, targetRoot: {}, type: {}, max_depth: {}, sequence_size: {}, shuffle: {}".format(
        sourceRoot,targetRoot,type, max_depth, sequence_size, shuffle
    ))

    samples = glob.glob(sourceRoot + type +'/*.ply')
    samples = sorted(samples, key=lambda name: name) 
    for filename in samples:
        filenamePrefix = filename.split("/")[-1].split(".")[0]  
        targetFileName = targetRoot + type + "/" + filenamePrefix + ".npy" # 00_000000.npy
        pc = np.asarray(o3d.io.read_point_cloud(filename).points)
        nowOctree = myutils.octree_construction(pc,max_depth).root
        nowData = generateDataFromOctree_transformer_cpp(nowOctree,sequence_size,shuffle)

        sequenceNum = len(nowData[0])
        for i in range(sequenceNum):
            if(len(nowData[0][i]) != len(nowData[1][i])):
                raise Exception("length not identical")
            if(len(nowData[1][i]) < sequence_size):
                for j in range(sequence_size - len(nowData[1][i])):
                    nowData[0][i].append((0,0,0,0,0,0)) #  x,y,z,depth,index,occupancy
                    nowData[1][i].append(0)
        np.save(targetFileName,np.array(nowData,dtype=object))


if __name__ == '__main__':
    _max_depth = 8
    _sequence_size = 512
    _sourceRoot = "YOUR_PLY_ROOT/" # ply root
    _targetRoot = "YOUR_NPY_ROOT/sequence{}_octree{}_split/".format(_sequence_size,_max_depth) # npy root
    _shuffle = True
    types = ["test"] # ["train","val","test"]
    for _type in types:
        start = time.time()
        generateOctreeNpy(sourceRoot=_sourceRoot,targetRoot=_targetRoot,type=_type,max_depth=_max_depth,sequence_size=_sequence_size,shuffle=_shuffle)
        end = time.time()
        print("time cost: ",end-start)


