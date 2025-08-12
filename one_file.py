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

import pandas as pd
def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)

from scipy.spatial import KDTree
def distChamfer(f1, f2, scale=1.0):
    f1 = f1/ scale
    f2 = f2/ scale
    tree = KDTree(f1, compact_nodes=False)
    d1, _ = tree.query(f2, k=1, workers=-1, eps=0)
    tree = KDTree(f2, compact_nodes=False)
    d2, _ = tree.query(f1, k=1, workers=-1, eps=0)
    return max(d1.mean(), d2.mean())

def parse_psnr_output(output_str):
    c = output_str.splitlines()
    for i in range(len(c)):
        if c[i].startswith('3.'):
            d1 = float(c[i+2].split(' ')[-1])
            try:
                d2 = float(c[i+4].split(' ')[-1])
            except Exception as e:
                d2 = 0.
            break
    return d1, d2

def process(input_f, dec_f=None, peak=59.7):
    filename_w_ext = os.path.split(input_f)[-1]
    dec_f = dec_f if dec_f is not None else os.path.join(output_path, filename_w_ext)
    peak = str(peak)
    cmd = f'./pc_error \
    -a {input_f} -b {dec_f} \
    -r {peak}'
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    d1_psnr, d2_psnr = parse_psnr_output(output.decode("utf-8"))
    
    return np.array([filename_w_ext, d1_psnr, d2_psnr])

# export LD_LIBRARY_PATH=~/.conda/envs/chunjie/lib:$LD_LIBRARY_PATH
print("start")
cuda_id = 7
device = torch.device(f"cuda:{cuda_id}")

with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        best_model=torch.load('./models/kitti_best.pth').to(device)

output_path = "./"

if not os.path.exists(output_path):
    os.mkdir(output_path)

print("start")

files = [
    ('/home/fengmj8/scj/PointSoupOrigin/data/SemanticKittiTestMini/50.ply', 59.7),
        #  ('/home/fengmj8/scj/PointSoupOrigin/data/FordFinal/1600.ply', 30000)
         ]
from thop import profile

depth = 12
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
start_mem = torch.cuda.memory_allocated()

for file_path, peak in files:
    total_flops = 0

    point_cloud = read_point_cloud(file_path)
    norm_point_cloud, min_, max_ =  normalize_data(point_cloud)
    root = octreebuild_BFS_uniform(norm_point_cloud, depth)
    bits, re_point_cloud, compress_time, total_flops = \
        octformer_encode_root(best_model, root, depth, cuda_id=cuda_id)
    bpp = (bits + 64) / len(point_cloud)
    filename_w_ext = os.path.split(file_path)[-1]
    decompressed_path = os.path.join(output_path, filename_w_ext)
    re_point_cloud = denormalize_data(re_point_cloud, min_, max_)
    save_point_cloud(re_point_cloud, decompressed_path)

    cd = distChamfer(point_cloud, re_point_cloud, scale=0.01)

    _, d1, d2 = process(file_path, decompressed_path, peak)
    save_point_cloud(re_point_cloud, output_path + \
                     f"OctFormer bpp {float(bpp):.2f} PSNR {float(d1):.1f}_" + filename_w_ext)

    print(f"{filename_w_ext} bpp {float(bpp):.2f} PSNR {float(d1):.1f}")

    os.remove(decompressed_path)
    print(f"total FLOPS: {total_flops/ (1024**3)} G")


peak_mem = torch.cuda.max_memory_allocated()
used_mem = peak_mem - start_mem
print(f"peak_mem: {peak_mem / (1024 ** 3)} G, used_mem: {used_mem / (1024 ** 3)} G")