from os import times
import os
from octAttention import TransformerModel
import torch
import time

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == '__main__':
    
    ntokens = 255 # the size of vocabulary
    ninp = 4*(128+4+6) # embedding dimension


    nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0 # the dropout value

    bptt = 8
    levelNumK = 4
    K = 1000
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Total params: ",total)
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).cuda()
    input = torch.zeros((bptt,1,levelNumK,3)).long().cuda()

    
    # encoding
    loop_time = 10
    all_sum = 0
    for i in range(loop_time):
        timeSum = 0
        nodeNum = 0
        while(True):
            start = time.time()
            pred = model(input,src_mask,[])
            end = time.time()
            nodeNum += bptt
            timeSum += end-start
            if(nodeNum > K):
                break
        all_sum += timeSum
    print("When N is {}，the encoding time for 1000 nodes is {}".format(bptt,all_sum/loop_time))

    # decoding
    loop_time = 10
    all_sum = 0

    for i in range(loop_time):
        timeSum = 0
        nodeNum = 0
        while(True):
            start = time.time()
            pred = model(input,src_mask,[])
            end = time.time()
            nodeNum += 1
            timeSum += end-start
            if(nodeNum > K):
                break
        all_sum += timeSum
    print("When N is {}，the decoding time for 1000 nodes is {}".format(bptt,all_sum/loop_time))
