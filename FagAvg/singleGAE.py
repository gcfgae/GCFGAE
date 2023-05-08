import torch
import time
import random
import argparse
import scipy.sparse as sp
from torch.nn import Parameter
from torch import optim
import numpy as np
from utils import single_load_data,single_preprocess_graph, sparse_mx_to_torch_sparse_tensor
from model import SingleGAE
from optimizer import loss_function_AE
from copy import deepcopy,copy


class SingleClient:
    def __init__(self, args, edge_path, feat_path, hidden_sizes):
        self.edge_path = edge_path
        self.feat_path = feat_path
        # self.real = isreal
        self.adj, self.feature,self.nodes_list = self.load_data()
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0]=self.feature.shape[1] #在没有通过一的初始全局节点时要加上
        self.single_model=SingleGAE(self.hidden_sizes,args).double()
        self.train_epoch=0
    def load_data(self):
        adj, features,node_list = single_load_data(self.edge_path, self.feat_path)

        return adj, features,node_list

    def prepare_data(self):
        self.n_nodes, self.feat_dim = self.feature.shape
        adj_train = self.adj

        self.adj_norm = single_preprocess_graph(self.adj)

        self.adj_label = adj_train + sp.eye(adj_train.shape[0])
        self.adj_label = torch.DoubleTensor(self.adj_label.toarray())
        self.pos_weight = (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

        self.norm = self.adj.shape[0] * self.adj.shape[0] / (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum() * 2)

        self.att_norm = self.feature.shape[1] * self.feature.shape[1] / (
                    self.feature.shape[1] * self.feature.shape[1] - self.feature.sum() * 2)

    def init_weight(self,model_dict):
        self.single_model.gc1.weight = deepcopy(Parameter(model_dict['gc1.weight']))
        self.single_model.gc2.weight = deepcopy(Parameter(model_dict['gc2.weight']))
        self.single_model.gc3.weight = deepcopy(Parameter(model_dict['gc3.weight']))


        
    def train_model(self,local_epoch):
        loss_last=0
        loss_num=0
        for l_epoch in range(local_epoch):  
            self.single_model.train()
            input1, support1, output1, hidden_vector1, input2, support2, output2, hidden_vector2, input3, support3, output3, hidden_vector3, recover_matrix = self.single_model(
                self.feature, self.adj_norm)
            client_loss = loss_function_AE(preds=recover_matrix, labels=self.adj_label,
                                           norm=self.norm, pos_weight=self.pos_weight)

            client_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
           

        self.train_epoch=l_epoch
        self.hidden_vextor=hidden_vector3
    
    
    def update_weight(self,global_wight_gc1, global_wight_gc2, global_wight_gc3):
        self.single_model.gc1.weight=deepcopy(Parameter(global_wight_gc1))
        self.single_model.gc2.weight=deepcopy(Parameter(global_wight_gc2))
        self.single_model.gc3.weight=deepcopy(Parameter(global_wight_gc3))
        self.optimizer = optim.Adam(self.single_model.parameters(), lr=0.01)
        


