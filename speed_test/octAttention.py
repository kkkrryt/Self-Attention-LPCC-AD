import math
import torch
import torch.nn as nn
import os
import datetime
import copy
# from networkTool import *
# from torch.utils.tensorboard import SummaryWriter

##########################

ntokens = 255 # the size of vocabulary
ninp = 4*(128+4+6) # embedding dimension


nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
batchSize = 32

MAX_OCTREE_LEVEL = 12


class SelfMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0.5):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  # 168//4=42  
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  # 
        self.mlpKey = nn.Linear(emsize, self.all_head_size) #MLP(168,168)
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)
        self.mlpValue = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)
        
    # Slice the output of mlpKQV to implement multi-head attention.
    def slice(self,x,dim):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size) # [batch_size, bptt, nhead, head_size] or [batch_size, bptt, levelNumK, nhead, head_size]
        x = x.view(*new_x_shape)
        if (dim == 3):
            x = x.permute(0, 2, 1, 3)
        elif (dim == 4):
            x = x.permute(0,1,3,2,4)
            assert 0
        return x

    #em.shape = [bptt,batch_size,emsize]  mask.shape=[bptt, bptt]
    def forward(self,em,mask):
        em = em.transpose(0,1).contiguous()  #[batch_size,bptt,...]
        Key = self.slice(self.mlpKey(em),em.dim())    # [batch_size, bptt, all_head_size] -> [batch_size,nhead,bptt,head_size]
        Query = self.slice(self.mlpQuery(em),em.dim()) #torch.Size([32, 4, 256, 42])
        Value = self.slice(self.mlpValue(em),em.dim())

        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)  #[batch_size,nhead,bptt,bptt] or [bs,bptt,nhead,levelNumK,levelNumK]
        attention_score = attention_score + mask #torch.Size([32, 4, 256, 256]) ,mask [[0,-inf,-inf,..],[0,0,-inf,...],[0,0,0,...]]
        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))

        context = torch.matmul(attention_map, Value)        # [batch_size, 4 nhead, bptt, 42 head_size] #torch.Size([32, 4, 256, 42])
        if (context.dim() == 4):
            context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, bptt, 4 nhead, 42 head_size]
        elif (context.dim()==5):
            context = context.permute(0, 1, 3, 2, 4).contiguous()  # [batch_size, bptt, levelNumK, 8, 64]
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0,1).contiguous()
        return context
 
class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.MultiAttention = SelfMultiheadAttention(emsize=ninp,nhead=nhead)
        self.linear1 = nn.Linear(ninp,nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid,ninp)

        self.norm1 = nn.LayerNorm(ninp, eps=1e-5) # It will affect parallel coding 
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src is the integration of leaf node and its ancestors.
    def forward(self, src, src_mask):
        src2 = self.MultiAttention(src,src_mask)  #Multi-head Attention
        src = self.dropout1(src2) + src
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))  #[batch_size,bptt,ninp] -> [batch_size,bptt,nhid] -> [batch_size,bptt,ninp]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])

    def forward(self,src,src_mask):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data )
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data= nn.init.xavier_normal_(self.decoder0.weight.data )
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data )
    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct = src[:,:,:,0] #oct[bptt,batchsize,FeatDim(levels)] [0~254]
        level = src[:,:,:,1]  # [0~12] 0 for padding data
        octant = src[:,:,:,2] # [0~8] 0 for padding data

        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        
        level -= torch.clip(level[:,:,-1:] - 10,0,None)# the max level in traning dataset is 10        
        torch.clip_(level,0,MAX_OCTREE_LEVEL) 
        aOct = self.encoder(oct.long()) #a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct,aLevel,aOctant),3)

        a = a.reshape((bptt,batch,-1)) 
        
        # src = self.ancestor_attention(a)
        src = a.reshape((bptt,a.shape[1],self.ninp))* math.sqrt(self.ninp)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder1(self.act(self.decoder0(output)))
        return output

######################################################################
# ``PositionalEncoding`` module 
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:] # this moves the feat(octant,level) of current node to lastrow,        
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]# which will be used as known feat
    return data[:,:,-levelNumK:,:], (target).long(),[]


