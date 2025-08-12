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

###########################
import logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # 设置日志级别
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
#########################

def normalize_data(data):
    normalized_input = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    return normalized_input, np.amin(data), np.amax(data)

def denormalize_data(data, min_, max_):
    normalized_input = data * (max_ - min_) + min_
    return normalized_input

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

def process(input_f):
    filename_w_ext = os.path.split(input_f)[-1]
    dec_f = os.path.join(save_path, filename_w_ext)
    peak = '59.70'
    cmd = f'./pc_error \
    -a {input_f} -b {dec_f} \
    -r {peak}'
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    d1_psnr, d2_psnr = parse_psnr_output(output.decode("utf-8"))
    
    return np.array([filename_w_ext, d1_psnr, d2_psnr])

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

from scipy.spatial import KDTree
def distChamfer(f1, f2, scale=1.0):
    f1 /= scale
    f2 /= scale
    tree = KDTree(f1, compact_nodes=False)
    d1, _ = tree.query(f2, k=1, workers=-1, eps=0)
    tree = KDTree(f2, compact_nodes=False)
    d2, _ = tree.query(f1, k=1, workers=-1, eps=0)
    return max(d1.mean(), d2.mean())

# export LD_LIBRARY_PATH=~/.conda/envs/chunjie/lib:$LD_LIBRARY_PATH
print("start")
path = "/home/fengmj8/scj/pointcloud_works/experiments/SemanticKITTI/test/SemanticKittiTestMini/*.ply"
device = torch.device("cuda:5")
save_path = "./Cache/"
files = np.array(glob(path, recursive=True))
print(len(files))

with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        best_model=torch.load('./models/kitti_best.pth').to(device)

# point_cloud = read_point_cloud(files[0])
# norm_point_cloud, min_, max_ =  normalize_data(point_cloud)
# root = octreebuild_BFS_uniform(norm_point_cloud, 11)
# rec = octree2pointcloud(root)
# print(point_cloud.shape, rec.shape)

# norm_point_cloud, min_, max_ =  normalize_data(point_cloud)
# root = octreebuild_BFS_uniform(norm_point_cloud, 9)
# rec = octree2pointcloud(root)
# print(point_cloud.shape, rec.shape)


for depth in range(11, 12):
    logger.info(f"########     depth {depth}     #################")
    print(f"########     depth {depth}     #################")
    avg_bpp = 0
    avg_CD = 0
    for file in tqdm(files):
        save_file = os.path.join(save_path, os.path.split(file)[-1])

        point_cloud = read_point_cloud(file)
        norm_point_cloud, min_, max_ =  normalize_data(point_cloud)
        root = octreebuild_BFS_uniform(norm_point_cloud, depth)
        bits, re_point_cloud, compress_time = octformer_encode_root(best_model, root, depth, cuda_id=2)
        bpp = (bits + 64) / len(point_cloud)

        avg_bpp += bpp

        re_point_cloud = denormalize_data(re_point_cloud, min_, max_)
        write_ply_data(save_file, re_point_cloud)
        cd = distChamfer(point_cloud, re_point_cloud, scale=0.001)
        avg_CD += cd
    stri = f'Avg. bpp: {avg_bpp / len(files)}'
    print(stri)
    logger.info(stri)
    stri = f'Avg. CD (cm): {avg_CD / len(files)}'
    print(stri)
    logger.info(stri)

    f_len = len(files)
    with Pool(8) as p:
        arr = list(tqdm(p.imap(process, files), total=f_len, file=sys.stderr, dynamic_ncols=True))

    arr = np.array(arr)
    fnames, p2pPSNRs, p2plainPSNRs = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)
        
    stri = f'Avg. D1 PSNR: {round(p2pPSNRs.mean(), 3)} Avg. D2 PSNR: {round(p2plainPSNRs.mean(), 3)}'
    print(stri)
    logger.info(stri)
