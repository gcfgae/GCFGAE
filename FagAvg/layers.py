import argparse
import time
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
import math


class GraphConvolution(Module):


    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        torch.set_default_tensor_type(torch.DoubleTensor)
        input = F.dropout(input, self.dropout, self.training)
       
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output2 = self.act(output)
        return input,support,output,output2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


