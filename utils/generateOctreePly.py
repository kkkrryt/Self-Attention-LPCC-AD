'''
.ply -> octree -> .ply
'''
from tkinter import E
import numpy as np
import sys
sys.path.append("../")
import pybind.myutils as myutils
import open3d as o3d
import glob
from tqdm import tqdm

def pointcloud2Octree(filename,depth):
    pc = o3d.io.read_point_cloud(filename)
    xyz = np.asarray(pc.points)
    octree = myutils.octree_construction(xyz,depth).root
    return octree

def generateOctreePlyForOneFile(filename, plyfilename, depth):
    if(depth > 0 ):
        root = pointcloud2Octree(filename, depth)
        oct_p = myutils.octree2pointcloud(root)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(oct_p)
        o3d.io.write_point_cloud(plyfilename,pcd1,write_ascii=True)
    else:
        xyz = np.load(filename).astype(np.float32)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(plyfilename,pcd1,write_ascii=True)


if(__name__ == '__main__'):
    # kitti part
    depth = 13
    inFilename = "../plys/13_000000.ply"
    outFilename = "../plys/13_000000_depth{}.ply".format(depth-1)
    generateOctreePlyForOneFile(inFilename,outFilename, depth)



