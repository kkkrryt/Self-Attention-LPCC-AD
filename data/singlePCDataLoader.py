import os
from random import randint
import random
import sys

from torch.nn.modules.activation import ReLU
sys.path.append("../")
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.preprocess import generateDataFromOctree_transformer_cpp
import time
# import pybind.myutils as myutils
import open3d as o3d

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
    
class singleOctDataLoader(Dataset):
    def __init__(self, nowOctree, max_depth = 13, sequence_size=128,shuffle =False):
        self.nowData = generateDataFromOctree_transformer_cpp(nowOctree,sequence_size,shuffle)
        self.dataNum = len(self.nowData[0])
    def __getitem__(self, idx):
        sequence = self.nowData[0][idx]
        label = self.nowData[1][idx]
        return torch.tensor(sequence),torch.tensor(label)
    def __len__(self):
        return self.dataNum
