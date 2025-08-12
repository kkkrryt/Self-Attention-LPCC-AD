import open3d as o3d
import numpy as np
from open3d import *
import argparse

'''
point cloud visualization script
usage: python vis.py -f <filename>
'''

def vispc_ply(filename):
    pc = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pc])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./vis.py")
    parser.add_argument(
        '--filename', '-f',
        type=str,
        required=True,
        help='point cloud file',
    )
    FLAGS, unparsed = parser.parse_known_args()

    
    filename = FLAGS.filename
    vispc_ply(filename)     
