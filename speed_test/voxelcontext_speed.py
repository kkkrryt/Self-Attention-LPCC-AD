from os import times
import sys
sys.path.append("../")
from VoxelContextNet import VoxelContextNet
import torch
import time

if __name__ == '__main__':
    K_Node = 1000
    _voxel_size = 9
    model = VoxelContextNet(voxel_size = _voxel_size).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Total params: ",total)
    model.eval()
    _voxel = torch.zeros((1,1,_voxel_size,_voxel_size,_voxel_size)).cuda()
    _feature = torch.zeros((1,6)).cuda()
    loop_time = 10
    all_sum = 0
    for i in range(loop_time):
        timeSum = 0
        nodeNum = 0
        while(True):
            start = time.time()
            pred = model(_voxel,_feature)
            end = time.time()
            nodeNum += 1
            timeSum += end-start
            if(nodeNum > K_Node):
                break
        all_sum += timeSum
    print("When voxel size is {}，the encoding/decoding time for 1000 nodes is {}".format(_voxel_size,all_sum/loop_time))
    
    