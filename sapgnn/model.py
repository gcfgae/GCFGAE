import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import os

from layers import GCNConv

from abc import ABC
import numpy as np
class GAE_InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(GAE_InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z,z.t()))
        return adj

class SingleGAE(nn.Module, ABC):
    def __init__(self,hidden_sizes,args):

        super(SingleGAE, self).__init__()
        self.gc1=GCNConv(hidden_sizes[0], hidden_sizes[1],act=nn.Tanh()).cuda()
        self.gc2=GCNConv(hidden_sizes[1], hidden_sizes[2],act=lambda x: x).cuda()
        self.gc3=GCNConv(hidden_sizes[2], hidden_sizes[3],act=lambda x: x).cuda()
        self.dc= GAE_InnerProductDecoder(args.dropout, act=lambda x: x) 
        self.hidden_sizes=hidden_sizes
        self.local_output = []

    def forward(self, g, f):

        
        hidden_vector1,m1=self.gc1(g,f)
        hidden_vector2,m2=self.gc2(g,hidden_vector1)
        hidden_vector3,m3=self.gc3(g,hidden_vector2)
        hidden_vector3.requires_grad_(True)
        return hidden_vector1,hidden_vector3,self.dc(hidden_vector3),hidden_vector2,m3
        

