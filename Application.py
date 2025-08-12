import numpy as np
from OctFormerEntropy import octreebuild_BFS_uniform, octformer_encode_root, octree2pointcloud
from glob import glob
from pyntcloud import PyntCloud
import os
from tqdm import tqdm
import subprocess
from multiprocessing import Pool
import sys
import warnings
import torch
from pytorch3d.loss import chamfer_distance
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pyntcloud.io.ply")


def normalize_data(data):
    normalized_input = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    return normalized_input, np.amin(data), np.amax(data)

def denormalize_data(data, min_, max_):
    normalized_input = data * (max_ - min_) + min_
    return normalized_input

def read_point_cloud(filepath):
    pc = PyntCloud.from_file(filepath)
    pc = np.array(pc.points, dtype=np.float32)[:, :3]
    assert not np.isnan(pc).any(), f"nan: {np.sum(np.isnan(pc))}"
    return pc

def write_ply_data(filename, points, attributeName=[], attriType=[]):
    """
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    """
    # if os.path.exists(filename):
    #   os.system('rm '+filename)
    if type(points) is list:
        points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1] >= (attrNum + 3)

    if os.path.dirname(filename) != "" and not os.path.exists(
        os.path.dirname(filename)
    ):
        os.makedirs(os.path.dirname(filename))

    plyheader = (
        ["ply\n", "format ascii 1.0\n"]
        + ["element vertex " + str(points.shape[0]) + "\n"]
        + ["property float x\n", "property float y\n", "property float z\n"]
    )
    for aN, attrName in enumerate(attributeName):
        plyheader.extend(["property " + attriType[aN] + " " + attrName + "\n"])
    plyheader.append("end_header")
    typeList = {"uint16": "%d", "float": "float", "uchar": "%d"}

    np.savetxt(
        filename,
        points,
        newline="\n",
        fmt=["%f", "%f", "%f"] + [typeList[t] for t in attriType],
        header="".join(plyheader),
        comments="",
    )
    return

import pandas as pd
def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)


# export LD_LIBRARY_PATH=~/.conda/envs/chunjie/lib:$LD_LIBRARY_PATH
print("start")
cuda_id = 7
device = torch.device(f"cuda:{cuda_id}")

with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        best_model=torch.load('./models/kitti_best.pth').to(device)

print("start")

input_globs = '/home/fengmj8/scj/Application/*.ply'
output_path = "./App/"

if not os.path.exists(output_path):
    os.mkdir(output_path)

files = np.array(glob(input_globs, recursive=True))
files = np.sort(files)

for depth, level in zip([11, 10, 9, 8, 7], [1, 2, 3, 4, 5]):
    avg_bpp = 0
    count = 0
    tq = tqdm(files, ncols=150)
    for file_path in tq:
        point_cloud = read_point_cloud(file_path)
        norm_point_cloud, min_, max_ =  normalize_data(point_cloud)
        root = octreebuild_BFS_uniform(norm_point_cloud, depth)
        bits, re_point_cloud, compress_time = octformer_encode_root(best_model, root, depth, cuda_id=cuda_id)
        bpp = (bits + 64) / len(point_cloud)
        avg_bpp += bpp
        count += 1

        filename_w_ext = os.path.split(file_path)[-1]
        if not os.path.exists(os.path.join(output_path, f"{level}/")):
            os.mkdir(os.path.join(output_path, f"{level}/"))
        decompressed_path = os.path.join(output_path, f"{level}/", filename_w_ext)

        re_point_cloud = denormalize_data(re_point_cloud, min_, max_)
        save_point_cloud(re_point_cloud, decompressed_path)
        tq.set_description(f"depth {depth} AVG bpp {avg_bpp/count:.3f}")