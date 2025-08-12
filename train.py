import csv
from glob import escape
from os import lchown, truncate
from sys import api_version
import time
from torch.nn.modules.transformer import TransformerEncoder
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from options import prepare_train_args
from utils.logger import Logger
from data.scannet_dataset_transformer_mem import ScanNetv2Octree_transformer_mem
from data.kitti_dataset_transformer import KittiOctree_transformer
from models.OctFormer import OctFormer
from matplotlib import pyplot as plt
import os


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        self.useDecisionTree = False
        self.useMLP = False
        self.useTransformer = True
        self.useFocalLoss = False

        _hidden = 128

       # **************** load/create model ******************
        if(self.args.load_model_path != 'no model'):
            self.model = torch.load(self.args.load_model_path).cuda()
            print("load model: ",self.args.load_model_path)
        else:
            self.model = OctFormer(sequence_size=self.args.sequence_size, 
                                        dropout_rate=self.args.dropout,
                                        hidden = self.args.hidden,
                                        nhead = self.args.nhead,
                                        num_layer = self.args.num_layer,
                                        absolute_pos = self.args.use_absolute_pos,
                                        OctLeff =  self.args.use_OctLeFF,
                                        OctPEG = self.args.use_OctPEG,
                                        ).cuda()
         
        # **************** DataLoader ******************
        if(self.args.dataset == 'scannet'):
            self.train_loader = DataLoader(ScanNetv2Octree_transformer_mem(type="train",sequence_size=self.args.sequence_size,max_depth=self.args.tree_depth), batch_size=self.args.batch_size, shuffle=True, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)
            self.val_loader = DataLoader(ScanNetv2Octree_transformer_mem(type="val",sequence_size=self.args.sequence_size,max_depth=self.args.tree_depth), batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_fn_transformer)
        
        elif(self.args.dataset == 'kitti'):
            self.train_loader = DataLoader(KittiOctree_transformer(sequence_size=self.args.sequence_size, type="train"),batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=0,)
            self.val_loader = DataLoader(KittiOctree_transformer(sequence_size=self.args.sequence_size, type="val"),batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=0,)

        else:
            raise Exception("unrecognized dataset")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)
        self.batch_size = self.args.batch_size
        print("tree_depth{}, sequence_size{}, batch_size{}, lr{}, weight_decay:{}, dropout:{}, hidden:{}".format(self.args.tree_depth,
                                                                                                self.args.sequence_size,
                                                                                                self.args.batch_size,
                                                                                                self.args.lr,
                                                                                                self.args.weight_decay,
                                                                                                self.args.dropout,
                                                                                                self.args.hidden
                                                                                                ))
        self.minValLoss = 999999
        self.bestModel = None

    def train(self):
        train_losses = []
        val_losses = []
        val_bpps = []
        epochs = []
        val_epochs = []

        for epoch in range(0,self.args.epochs+1):
            print("############################epoch{}############################".format(epoch))
            # train for one epoch
            train_start = time.time()
            train_loss = self.train_per_epoch(epoch)
            train_end = time.time()
            print("epoch {}, time cost: {} s".format(epoch,train_end-train_start))
            train_losses.append(train_loss)
            epochs.append(epoch)
            if(epoch % self.args.val_freq == 0):
                # val for one epoch
                val_start = time.time()
                val_loss,val_bpp = self.val_per_epoch(epoch)
                val_end = time.time()
                print("epoch {} time cost:{} s".format(epoch,val_end-val_start))
                val_losses.append(val_loss.cpu())
                val_bpps.append(val_bpp.cpu())
                val_epochs.append(epoch)
            
           
        img_root = self.args.save_dir + "images/"
        if(os.path.exists(img_root) == False):
            os.mkdir(img_root)
        plt.xlabel('epochs')
        plt.ylabel('train_loss')
        plt.plot(epochs,train_losses)
        plt.savefig(img_root+"train_loss")

        plt.clf()

        plt.xlabel('epochs')
        plt.ylabel('val_loss')
        plt.plot(val_epochs,val_losses)
        plt.savefig(img_root+"val_loss")

        plt.clf()

        plt.xlabel('epochs')
        plt.ylabel('val_bpp')
        plt.plot(val_epochs,val_bpps)
        plt.savefig(img_root+"val_bpp")

        print("img save to...",img_root)
    
    
    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        overall_acc = 0
        mean_batch_loss = 0
        overall_bitrate = 0
        batch_num = 0
        for i, data in enumerate(self.train_loader):
            batch_num = batch_num + 1
            self.optimizer.zero_grad()
            h, adj, label, pred = self.step(data)
            h = h.to(torch.float32)
            adj = adj.to(torch.float32)
            pred = pred.to(torch.float32)
            loss = self.compute_loss_transformer(pred,label.long())        
            loss.backward()
            mean_batch_loss = mean_batch_loss + loss.item()
            self.optimizer.step()

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
        print("train: mean_batch_loss for epoch{} is {}".format(
            epoch, 
             mean_batch_loss/batch_num,
           ))
        return mean_batch_loss/batch_num

    def val_per_epoch(self, epoch):
        self.model.eval()
        overall_acc = 0
        mean_batch_loss = 0
        batch_num = 0
        bpp = 0
        for i, data in enumerate(self.val_loader):
            batch_num = batch_num + 1
            with torch.no_grad():
                h, adj, label, pred = self.step(data)
            h = h.to(torch.float32)
            adj = adj.to(torch.float32)
            pred = pred.to(torch.float32)
            # compute loss
            loss = self.compute_loss_transformer(pred,label.long())
            bitrate = self.compute_bitrate_transformer(pred,label)
            bpp = bpp + bitrate/7678758  
            mean_batch_loss = mean_batch_loss + loss

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Val: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
        print("Val: mean_batch_loss for epoch {} is {}, bpp is {} ".format(epoch, mean_batch_loss/batch_num,bpp)) #  overall_acc/batch_num, 
        if(mean_batch_loss/batch_num < self.minValLoss):
            print("epoch {}，perform best, loss: {}， bpp: {}".format(epoch,mean_batch_loss/batch_num,bpp))
            self.minValLoss = mean_batch_loss/batch_num
            self.bestModel = self.model
            ## save best model
            print("########################################")
            save_path = self.args.save_dir + "bestmodel.pth"
            torch.save(self.bestModel,save_path)
            print("model save to...",save_path)

        return mean_batch_loss/batch_num,bpp


    def step(self, data):
        (h,label) = data
        h = Variable(h).cuda().to(torch.float32)
        label = Variable(label).cuda().to(torch.float32)
        pred = self.model(h)
        return h,torch.zeros(1),label,pred  # torch.zeros(1) is a trick


    def compute_loss_transformer(self,pred,gt):
        loss = 0
        for i in range(pred.shape[1]):
            if(self.useFocalLoss):
                loss += self.focalLoss(pred[:,i,:], gt[:,i])
            else:
                loss += torch.nn.functional.cross_entropy(pred[:,i,:], gt[:,i])
        return loss

    def compute_bitrate_transformer(self,pred,gt):
        batch_size = pred.shape[0]
        sequence_size = pred.shape[1]
        logit = F.softmax(pred,dim=2)

        one_hot_gt = torch.zeros(batch_size,sequence_size,256)
        one_hot_gt = one_hot_gt.scatter_(dim=2,index=gt.reshape(batch_size,sequence_size,1).data.long().cpu(),value=1)
        ans = torch.mul(one_hot_gt.cuda(),logit.cuda())
        ans[ans == 0] = 1
        bits = -torch.log2(ans).sum()
        return bits

def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()