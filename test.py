import torch
import numpy as np
import open3d as o3d
from torch._C import _rpc_init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.scannet_dataset_transformer_mem import ScanNetv2Octree_transformer_mem
from data.kitti_dataset_transformer import KittiOctree_transformer
import torch.nn.functional as F
from tqdm import tqdm
import time

def collate_fn_transformer(batch):
    batch = list(zip(*batch))
    hs = batch[0]
    labels = batch[1]

    new_hs = []
    new_labels = []

    maxSequenceSize = 0

    for h in hs:
        maxSequenceSize = max(maxSequenceSize, len(h))

    batch_size = len(hs)

    for index in range(batch_size):
        h = hs[index]
        curLen = len(h)
        new_hs.append(torch.cat((hs[index],torch.zeros(maxSequenceSize-curLen,6)),0))
        new_labels.append(torch.cat((labels[index],torch.zeros(maxSequenceSize-curLen))))
            

    new_hs = torch.cat(new_hs,dim=0).reshape(batch_size, maxSequenceSize, 6)
    new_labels = torch.cat(new_labels,dim=0).reshape(batch_size, maxSequenceSize)


    del batch
    return new_hs, new_labels

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

if __name__ == '__main__':

    # ******************************************************************
    # kitti part
    # ******************************************************************
    _tree_detphs = [13,12,11,10,9,8]
    result = {}
    best_model = torch.load("./models/kitti_best.pth")
    best_model.eval()
    for _tree_depth in _tree_detphs:
        test_batch_size = 32
        _sequenze_size = 512
        test_loader = DataLoader(KittiOctree_transformer(type="test",sequence_size=_sequenze_size,max_depth=_tree_depth), batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)
        bpp = 0
        start = time.time()
        for i, data in enumerate(test_loader):
            (h,label) = data
            h = Variable(h).cuda().to(torch.float32)
            label = Variable(label).cuda().to(torch.float32)
            pred = best_model(h)
            bitrate = compute_bitrate_transformer(pred,label)
            bpp = bpp + bitrate.data/2449921930  # total point cloud num
        end = time.time()
        print("time cost: ",end-start)
        result[_tree_depth] = bpp
        print("sequence_size:{}, depth:{}, bpp:{}".format(_sequenze_size,_tree_depth,bpp))
    print(result)

    # ******************************************************************
    # scannet part
    # ******************************************************************
    # _tree_detphs = [10,9,8,7,6] # 
    # result = {}
    # best_model = torch.load("./models/scannet_best.pth")
    # best_model.eval()
    # for _tree_depth in _tree_detphs:
    #     test_batch_size = 32
    #     _sequenze_size = 1024
    #     test_loader = DataLoader(ScanNetv2Octree_transformer_mem(type="test",sequence_size=_sequenze_size,max_depth=_tree_depth), batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)

    #     bpp = 0
    #     start = time.time()
    #     for i, data in enumerate(test_loader):
    #         (h,label) = data
    #         h = Variable(h).cuda().to(torch.float32)
    #         label = Variable(label).cuda().to(torch.float32)
    #         pred = best_model(h)
    #         bitrate = compute_bitrate_transformer(pred,label)
    #         bpp = bpp + bitrate.data/15487852  # total point cloud num
    #     print("time cost: ",end-start)
    #     result[_tree_depth] = bpp
    #     print("sequence_size:{}, depth:{}, bpp:{}".format(_sequenze_size,_tree_depth,bpp))
    # print(result)
