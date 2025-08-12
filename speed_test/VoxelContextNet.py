import numpy as np
import torch
from torch.jit import Error
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("../")

class VoxelContextNet(nn.Module):
    def __init__(self,voxel_size):
        super(VoxelContextNet,self).__init__()
        self.voxel_size =voxel_size
        self.conv3d1 = nn.Conv3d(1,32,kernel_size=(3,3,3),padding=1)
        self.conv3d2 = nn.Conv3d(32,64,kernel_size=(3,3,3),padding=1)
        self.conv3d3 = nn.Conv3d(64,128,kernel_size=(3,3,3))
        self.conv3d4 = nn.Conv3d(128,128,kernel_size=(3,3,3))
        if(voxel_size == 5):
            self.fc1 = nn.Linear(134,192)
        elif(voxel_size == 7):
            self.fc1 = nn.Linear(3462,192)
        elif(voxel_size == 9):
            self.fc1 = nn.Linear(16006,192)
        elif(voxel_size == 11):
            self.fc1 = nn.Linear(43910,192)
        else:
            raise Error("unexpected voxel size!")
        self.fc2 = nn.Linear(192,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
    
    def forward(self, voxels,features):
        out = self.conv3d1(voxels)
        out = self.conv3d2(out)
        out = self.conv3d3(out)
        out = self.conv3d4(out)
        out = out.reshape(out.shape[0],-1) # flatten
        out = torch.cat([out,features],dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


if __name__ == '__main__':
    _voxel_size = 11
    model = VoxelContextNet(_voxel_size)
    total = sum([param.nelement() for param in model.parameters()])
    print("Total params: ",total)
    torch.save(model,"vcn.pth")
    _voxel = torch.zeros(256,1,_voxel_size,_voxel_size,_voxel_size)
    _feature = torch.zeros(256,6)
    out = model(_voxel,_feature)
    print(out.shape)


