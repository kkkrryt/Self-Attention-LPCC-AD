import numpy as np
import sys
import copy
import open3d as o3d
import time
import pandas as pd
from typing import List, Tuple, Dict

from data.singlePCDataLoader import singleOctDataLoader,collate_fn_transformer
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader

import warnings

import matplotlib.pyplot as plt


import numba
sys.path.append("./NumpyAc")
import numpyAc


# 节点，构成OCtree的基本元素
class Octantids:
    def __init__(self, children, center, extent, depth:int, is_leaf:bool):
        self.children = children  # 子节点
        self.center = center  # 正方体的中心点
        self.extent = extent  # 正方体的边长一半
        self.is_leaf = is_leaf  # 是否叶节点
        self.depth = depth  # 节点的深度
        self.octant = 0  # octant，代表八个子节点内是否有点，0-255（即00000000-11111111）


    # 功能：打印节点信息
    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'octant: ' + str(self.octant)
        return output

def octreebuild_BFS_uniform(db_np:np.ndarray,max_layer:int,center_mean=False,new_center=None):

    if len(db_np)==0:
       return None,np.empty([0,3])     

    N, dim = db_np.shape[0], db_np.shape[1]

    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    if center_mean:
        db_center = np.mean(db_np, axis=0)
    else:
        db_center = (db_np_max+db_np_min)/2
    if new_center is not None:
        db_center = new_center
    # db_center=np.array([0.5,0.5,0.5])
    # db_extent = 0.5

    db_depth = 1
    point_indices=list(range(N))
    root = Octantids([None for i in range(8)], db_center, db_extent, db_depth, is_leaf=False)
    pointbuildinginfo=[point_indices]
    leafnodeList=[root]
    # respointlist=[]



    # 功能：对八叉树的下一层进行划分建树并提取剩余点
    def octree_split_nextlayer(lastlayer=False):
        nonlocal leafnodeList,pointbuildinginfo,db_np
        point_indices=pointbuildinginfo.pop(0)
        node = leafnodeList.pop(0)
        center=node.center
        extent=node.extent
        depth=node.depth
        node.is_leaf=False
        children_point_indices = [[] for i in range(8)]
        for point_idx in point_indices:
            point_db = db_np[point_idx]
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)
        # count octant
        for i in range(8):
            if (len(children_point_indices[i]) > 0):
                node.octant += 2 ** (7 - i)

        # create children
        factor = [-0.5, 0.5]
        for i in range(8):
            if len(children_point_indices[i])>0:
                child_center_x = center[0] + factor[(i & 1) > 0] * extent
                child_center_y = center[1] + factor[(i & 2) > 0] * extent
                child_center_z = center[2] + factor[(i & 4) > 0] * extent
                child_extent = 0.5 * extent
                child_center = np.asarray([child_center_x, child_center_y, child_center_z])
                node.children[i] = Octantids([None for i in range(8)], child_center, child_extent, depth+1, is_leaf=True)            
                pointbuildinginfo.append(children_point_indices[i])
                leafnodeList.append(node.children[i])
                # if(lastlayer):
                #     if len(children_point_indices[i])>1:
                #         resindex = copy.deepcopy(children_point_indices[i])
                #         del resindex[np.random.randint(len(children_point_indices[i]))]
                #         childpoint_db = db_np[resindex]
                #         respointlist.append(childpoint_db)


    current_layer=1
    while current_layer<max_layer:
        extract_flag=(current_layer == max_layer-1)
        NodeNum=len(leafnodeList)            
        for _ in range(NodeNum):
            octree_split_nextlayer(extract_flag)
        current_layer+=1
    
    # if len(respointlist)>0:
    #     respoints = np.concatenate(respointlist)
    # else:
    #     respoints = np.empty((0,3))

    return root

def nodenum_count(root):
    nodecount=[1]
    current_layer=1
    nodestack=[root]
    while len(nodestack)!=0:
        node = nodestack.pop(0)
        for child in node.children:
            if child!=None:
                nodestack.append(child)
                if child.depth!=current_layer:
                    current_layer+=1
                    nodecount.append(0)
                nodecount[current_layer-1]+=1
    return nodecount


def compute_bitrate_transformer(pred,gt):
    batch_size = pred.shape[0]
    sequence_size = pred.shape[1]
    logit = F.softmax(pred,dim=2)
    one_hot_gt = torch.zeros(batch_size,sequence_size,256)
    one_hot_gt = one_hot_gt.scatter_(dim=2,index=gt.reshape(batch_size,sequence_size,1).data.long().cpu(),value=1)
    ans = torch.mul(one_hot_gt.cuda(),logit.cuda())
    ans[ans == 0] = 1
    bits = -torch.log2(ans).sum()
    return bits


def fetch_octantdata(pred,l):
    batch_size,sequence_size,dim = pred.shape
    logit = F.softmax(pred,dim=2)
    pdf=logit.detach().cpu().numpy().reshape(batch_size*sequence_size,dim)
    sym=l.detach().cpu().numpy().reshape(batch_size*sequence_size)
    return pdf,sym

def compute_real_bitrate(pdf,sym,binpath):
    codec = numpyAc.arithmeticCoding()
    byte_stream,real_bits = codec.encode(pdf, sym, binpath)
    return byte_stream,real_bits

def octree2pointcloud(root):
    if root is None:
        return np.empty([0,3])
    else:      
        points = []
        def leaf_DFS(root):
            if root is None:
                return None
            else:
                if root.is_leaf:
                    points.append(root.center)
                    return None
                else:
                    for i in range(8):
                        leaf_DFS(root.children[i])
                    return None
        leaf_DFS(root)
        return np.asarray(points)
from thop import profile
# 基于八叉树的octformer压缩(输入基于根节点)
def octformer_encode_root(best_model, root, firstlayer=10,cuda_id:int=2):

    if root is None:
        return 0,np.array([]),0

    torch.cuda.set_device(cuda_id)
    begin=time.time()
   
    seq_size=512

    NaivePC3=octree2pointcloud(root)

    test_loader = DataLoader(singleOctDataLoader(nowOctree=root,sequence_size=seq_size,max_depth=firstlayer,shuffle=False), batch_size=32, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)   
    octants_num=nodenum_count(root)
    # bits1 = 0
    pdflist=[]
    symlist=[]

    total_flops = 0
    
    for i, data in enumerate(test_loader):
        (h,label) = data
        h = Variable(h).to(torch.float32).cuda(cuda_id)
        label = Variable(label).to(torch.float32).cuda(cuda_id)

        flops, params = profile(best_model,(h, ))
        total_flops += flops

        pred = best_model(h)
        # bitrate = compute_bitrate_transformer(pred,label)
        # bits1 = bits1 + bitrate.data # point cloud num
        pdf,sym=fetch_octantdata(pred,label)
        pdflist.append(pdf)
        symlist.append(sym)
        batch_seq=h.shape[1]
        
    # print('len of loader',len(test_loader),sum(octants_num))
    end_deepshannon=time.time()
    pdf_np=np.concatenate(pdflist,axis=0)
    sym_np=np.concatenate(symlist,axis=0)

    ind_current=0
    ind_layer=1
    pdf_dict={}
    sym_dict={}
    while ind_layer <firstlayer:
        octnum=octants_num[ind_layer-1]
        sym_dict[ind_layer]=sym_np[ind_current:ind_current+octnum].astype('int16')
        pdf_dict[ind_layer]=pdf_np[ind_current:ind_current+octnum,:]
        ind_current=ind_current+octnum
        while ind_current%batch_seq!=0:
            ind_current+=1
        ind_layer+=1
    total_rbits=0
    for i in range(1,firstlayer):
        _,rbits=compute_real_bitrate(pdf_dict[i],sym_dict[i],'./octantbins/'+str(i)+'out.b')
        total_rbits+=rbits
    
    # shannon_bits=bits1
    real_bits=total_rbits


    end=time.time()

    # print('基于root建树点数:',len(NaivePC3),'建树时间:',end_treebuild-begin,'推理时间:',end_deepshannon-end_treebuild,'算术编码时间',end-end_deepshannon)

    # return real bits, reconstructed point cloud from octree, compression time
    return real_bits,NaivePC3,end-begin, total_flops