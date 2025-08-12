import open3d as o3d
import numpy as np 
import os, time
import pandas as pd
import subprocess
import glob
from tqdm import tqdm

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, res, exe_path, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
               "h.       1(p2point)", "h.,PSNR  1(p2point)" ]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)", 
               "h.       2(p2point)", "h.,PSNR  2(p2point)" ]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", 
               "h.        (p2point)", "h.,PSNR   (p2point)" ]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF + haders_p2plane

    command = str(exe_path+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' -n '+infile1+
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(res))

    if normal:
      headers += haders_p2plane
      command = str(command + ' -n ' + infile1)

    results = {}
   
    start = time.time()
    subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c=subp.stdout.readline() 
    # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])


def compute_metrics_for_onefile(or_ply_path, rc_ply_path, exe_path):
    ori_pc = o3d.io.read_point_cloud(or_ply_path)
    ori_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # compute normal
    o3d.io.write_point_cloud("./temp_pc_normal/temp.ply",ori_pc,write_ascii=True)
    # double to float
    lines = open("./temp_pc_normal/temp.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("./temp_pc_normal/temp.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()
    # cal metrics
    results = pc_error("./temp_pc_normal/temp.ply", rc_ply_path, res=1, exe_path=exe_path,normal=True)
    d1_psnr = results["mseF,PSNR (p2point)"][0]
    d2_psnr = results["mseF,PSNR (p2plane)"][0]
    d1_mse = results["mseF      (p2point)"][0]
    res = {"d1_psnr":d1_psnr, "d2_psnr":d2_psnr, "d1_mse":d1_mse}
    return res

def compute_metrics_for_foler(or_ply_folder, rc_ply_foler, exe_path):
    d1_psnr_average = 0
    d2_psnr_average = 0
    d1_mse_average = 0
    or_files = glob.glob(or_ply_folder + "*.ply")
    rc_files = glob.glob(rc_ply_foler + "*.ply")
    or_files = sorted(or_files, key=lambda name: name)  
    rc_files = sorted(rc_files, key=lambda name: name)  
    for i in tqdm(range(len(or_files))):
        or_filename = or_files[i]
        rc_filename = rc_files[i]
        if(or_filename.split("/")[-1] != rc_filename.split("/")[-1]):
            print("filename not identical, please check")
            break
        result = compute_metrics_for_onefile(or_filename, rc_filename, exe_path)
        print("d1_psnr :{}, d2_psnr: {}".format(result['d1_psnr'],result['d2_psnr']))
        d1_psnr_average += result['d1_psnr']
        d2_psnr_average += result['d2_psnr']
        d1_mse_average += result['d1_mse']
    
    print("average d1_psnr {}, d2_psnr {}, mse {}".format(d1_psnr_average/len(or_files), d2_psnr_average/len(or_files), d1_mse_average/len(or_files)))
    return { "d1_psnr_avg": d1_psnr_average/len(or_files), 
             "d2_psnr_avg": d2_psnr_average/len(or_files),
             "d1_mse_avg": d1_mse_average/len(or_files)
            }


def normalize_point_cloud(pc:np.ndarray) ->np.ndarray:
    centroid = np.mean(pc, axis=0) # 求取点云的中心
    pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
    pc_normalized=(pc_normalized)/2+0.5 # 转化到(0,1)
    return pc_normalized

def normalize_data(data):
    normalized_input = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    #print((np.amax(data) - np.amin(data)))
    return normalized_input

def z_score_normalize(data):    
    mean = np.mean(data, axis=0)    
    std_dev = np.std(data, axis=0)    
    # normalized_data = (data - mean) / std_dev    
    normalized_data = (data - mean)
    return normalized_data

def together_npc(pc1:np.ndarray,pc2:np.ndarray):
    centroid = np.mean(pc1, axis=0) # 求取点云的中心
    pc1 = pc1 - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc1 ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized1 = pc1 / m # 依据长轴将点云归一化到 (-1, 1)
    pc_normalized1=(pc_normalized1)/2+0.5 # 转化到(0,1)

    pc2 = pc2 - centroid # 将点云中心置于原点 (0, 0, 0)
    pc_normalized2 = pc2 / m # 依据长轴将点云归一化到 (-1, 1)
    pc_normalized2=(pc_normalized2)/2+0.5 # 转化到(0,1)

    return pc_normalized1,pc_normalized2

def together_nd(data1,data2):
    normalized_input1 = (data1 - np.amin(data1)) / (np.amax(data1) - np.amin(data1))
    normalized_input2 = (data2 - np.amin(data1)) / (np.amax(data1) - np.amin(data1))
    #print((np.amax(data) - np.amin(data)))
    return normalized_input1,normalized_input2

def PSNR_count(points1:np.ndarray,points2:np.ndarray):

    # points1 = normalize_point_cloud(points1)
    # points2 = normalize_point_cloud(points2)
    points1 = normalize_data(points1)
    points2 = normalize_data(points2)

    # points1,points2=together_nd(points1,points2)

         
    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(points1.reshape(-1, 3))
    pt1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud("tempppt1.ply", pt1,write_ascii=True)

    pt2 = o3d.geometry.PointCloud()
    pt2.points = o3d.utility.Vector3dVector(points2.reshape(-1, 3))
    pt2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud("tempppt2.ply", pt2,write_ascii=True)

    # double to float
    lines = open("tempppt1.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("tempppt1.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()

    lines = open("tempppt2.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("tempppt2.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()

    # cal metrics
    results = pc_error("tempppt1.ply", "tempppt2.ply", res=1, exe_path='/home/eccblwxg/Desktop/OctFormer_release/metrics',normal=True)
    
    d1_psnr = results["mseF,PSNR (p2point)"][0]
    d2_psnr = results["mseF,PSNR (p2plane)"][0]
    d1_mse = results["mseF      (p2point)"][0]
    res = {"d1_psnr":d1_psnr, "d2_psnr":d2_psnr, "d1_mse":d1_mse}


    os.remove("tempppt1.ply")
    os.remove("tempppt2.ply")

    return d1_psnr,d2_psnr


def PSNR_count_peakValue(points1:np.ndarray,points2:np.ndarray,peakv):
         
    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(points1.reshape(-1, 3))
    pt1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud("tempppt1.ply", pt1,write_ascii=True)

    pt2 = o3d.geometry.PointCloud()
    pt2.points = o3d.utility.Vector3dVector(points2.reshape(-1, 3))
    pt2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud("tempppt2.ply", pt2,write_ascii=True)

    # double to float
    lines = open("tempppt1.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("tempppt1.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()

    lines = open("tempppt2.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("tempppt2.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()

    # cal metrics
    results = pc_error("tempppt1.ply", "tempppt2.ply", res=peakv, exe_path='/home/cuimy/fmj/projects/FOctFormer/OctFormer_release/metrics',normal=True)
    d1_psnr = results["mseF,PSNR (p2point)"][0]
    d2_psnr = results["mseF,PSNR (p2plane)"][0]
    d1_mse = results["mseF      (p2point)"][0]
    res = {"d1_psnr":d1_psnr, "d2_psnr":d2_psnr, "d1_mse":d1_mse}


    os.remove("tempppt1.ply")
    os.remove("tempppt2.ply")

    return d1_psnr,d2_psnr



if __name__ == '__main__':

    or_ply = "../plys/13_000000.ply"
    rc_ply = "../plys/13_000000_depth12.ply"
    print(compute_metrics_for_onefile(or_ply,rc_ply,"."))
