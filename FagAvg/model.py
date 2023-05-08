from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import os

from layers import GraphConvolution
from abc import ABC
import numpy as np

class SingleGAE(nn.Module, ABC):
    def __init__(self,hidden_sizes,args):
        
        super(SingleGAE, self).__init__()


        self.gc1=GraphConvolution(hidden_sizes[0], hidden_sizes[1], args.dropout, act=F.tanhshrink)

        self.gc2=GraphConvolution(hidden_sizes[1], hidden_sizes[2], args.dropout, act=lambda x: x)
        self.gc3=GraphConvolution(hidden_sizes[2], hidden_sizes[3], args.dropout, act=lambda x: x)

        self.dc= GAE_InnerProductDecoder(args.dropout, act=lambda x: x) #decoder

        self.hidden_sizes=hidden_sizes
        self.local_output = []

    def forward(self, x, adj):
    

        input1,support1,output1,hidden_vector1=self.gc1(x,adj)
        input2,support2,output2,hidden_vector2=self.gc2(hidden_vector1,adj)
        input3,support3,output3,hidden_vector3=self.gc3(hidden_vector2,adj)

        return input1,support1,output1,hidden_vector1,input2,support2,output2,hidden_vector2,input3,support3,output3,hidden_vector3,self.dc(hidden_vector3)


class GAE_InnerProductDecoder(nn.Module):


    def __init__(self, dropout, act=torch.sigmoid):
        super(GAE_InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Net(nn.Module):
    def __init__(self,input_dim,output_dim,args,act_func):
        super().__init__()
        self.gc=GraphConvolution(input_dim,output_dim, args.dropout, act=act_func)
    def forward(self,x,adj):
        torch.set_default_tensor_type(torch.DoubleTensor)
        input, support, output, hidden_vector = self.gc(x, adj) 
        return support,hidden_vector

