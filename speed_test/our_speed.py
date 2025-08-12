from os import times
import sys
sys.path.append("../")
from models.OctFormer import OctFormer
import torch
import time

if __name__ == '__main__':
    K_Node = 1000
    _sequence_size = 512
    model = OctFormer(sequence_size=_sequence_size,
                    dropout_rate=0.5,
                    hidden=256,
                    num_layer=6,
                    nhead=8,
                    absolute_pos="False",
                    OctLeff="True",
                    OctPEG="True",

                    ).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Total params: ",total)
    model.eval()
    h = torch.zeros((1,_sequence_size,6)).cuda()
    loop_time = 100
    all_sum = 0
    for i in range(loop_time):
        timeSum = 0
        nodeNum = 0
        while(True): 
            start = time.time()
            pred = model(h)
            end = time.time()
            nodeNum += _sequence_size
            timeSum += end-start
            if(nodeNum > K_Node):
                break
        all_sum += timeSum
    print("When N is {}，the encoding/decoding time for 1000 nodes is {}".format(_sequence_size,all_sum/loop_time))
    
    