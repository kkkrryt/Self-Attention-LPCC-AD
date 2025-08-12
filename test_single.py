'''
    Test the bpp of a single ply file using our method for compression.
    scannet: scene0011_00.ply
    kitti: 13_000000.ply
'''

import torch
import numpy as np
import open3d as o3d
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.singlePCDataLoader import singlePCDataLoader,collate_fn_transformer
import torch.nn.functional as F
from tqdm import tqdm
import time
import argparse


def compute_bitrate_transformer(pred,gt):
    batch_size = pred.shape[0]
    sequence_size = pred.shape[1]
    logit = F.softmax(pred,dim=2)

    # gt to one_hot  (batch_size, sequence_size, 256) 
    one_hot_gt = torch.zeros(batch_size,sequence_size,256)
    one_hot_gt = one_hot_gt.scatter_(dim=2,index=gt.reshape(batch_size,sequence_size,1).data.long().cpu(),value=1)
    ans = torch.mul(one_hot_gt.cuda(),logit.cuda())
    ans[ans == 0] = 1
    bits = -torch.log2(ans).sum()
    return bits

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./test_single.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='choose dataset',
    )
    FLAGS, unparsed = parser.parse_known_args()
    dataset = FLAGS.dataset
    if(dataset == 'kitti'):
        # KITTI
        _filename = "./plys/13_000000.ply"
        test_batch_size = 32
        _sequenze_size = 512
        _tree_depth = 13
        test_loader = DataLoader(singlePCDataLoader(filename=_filename,sequence_size=_sequenze_size,max_depth=_tree_depth,shuffle=False), batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)
        best_model = torch.load("./models/kitti_best.pth")
        bpp = 0
        for i, data in enumerate(test_loader):
            # break
            if(i % 20 == 0):
                print(i)
            (h,label) = data
            h = Variable(h).cuda().to(torch.float32)
            label = Variable(label).cuda().to(torch.float32)
            pred = best_model(h)
            bitrate = compute_bitrate_transformer(pred,label)
            bpp = bpp + bitrate.data/122785 # point cloud num
        print("The bpp of {} after compression is {}".format(_filename,bpp))
    elif(dataset == 'scannet'):
        # ScanNet
        _filename = "./plys/scene0011_00.ply"
        test_batch_size = 32
        _sequenze_size = 1024
        _tree_depth = 10
        test_loader = DataLoader(singlePCDataLoader(filename=_filename,sequence_size=_sequenze_size,max_depth=_tree_depth,shuffle=False), batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)
        best_model = torch.load("./models/scannet_best.pth")
        bpp = 0
        for i, data in enumerate(test_loader):
            # break
            if(i % 20 == 0):
                print(i)
            (h,label) = data
            h = Variable(h).cuda().to(torch.float32)
            label = Variable(label).cuda().to(torch.float32)
            pred = best_model(h)
            bitrate = compute_bitrate_transformer(pred,label)
            bpp = bpp + bitrate.data/50000 # point cloud num
        print("The bpp of {} after compression is {}".format(_filename,bpp))
    else:
        print("unkonwn dataset")