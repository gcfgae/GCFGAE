import time
from collections import Counter
from itertools import combinations_with_replacement
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd.functional import jacobian
from torch.nn import functional as F, Parameter
from functools import partial
from functorch import vmap, vjp, jacrev

import cupy as cp

from torch import optim
import torch.nn as nn
from functorch import jacrev

from Kmeans import adapt_dgl_hash__kmeans_onmi
from optimizer import loss_function_AE


class Server:
    def __init__(self,hidden_sizes,allnodes_num,alledges_num):
        self.participant_degree = {}
        self.allnodes_num = allnodes_num
        self.alledges_num = alledges_num
        self.overlapping_nodes_num = 0  
        self.norm = 0
        self.hidden_sizes = hidden_sizes
        self.linear1=nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1],bias=False).cuda()
        self.linear2=nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2],bias=False).cuda()
        self.linear3=nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3],bias=False).cuda()
        self.m1 = torch.zeros((self.allnodes_num, self.hidden_sizes[0]))
        self.m2 = torch.zeros((self.allnodes_num, self.hidden_sizes[1]))
        self.m3 = torch.zeros((self.allnodes_num, self.hidden_sizes[2]))
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    def set_parameter(self, model_dict):
        self.linear1.weight = Parameter(model_dict['gc1.weight'].t().cuda())
        self.linear2.weight = Parameter(model_dict['gc2.weight'].t().cuda())
        self.linear3.weight = Parameter(model_dict['gc3.weight'].t().cuda())

    def generate_all_h(self,h_list,all_participant_hashnode_list,layer):
        
        self.all_participant_hashnode_list=all_participant_hashnode_list
        self.node_h_dict={}
        
        if layer == 0:
            self.t1=h_list
        if layer == 1:
            self.t2=h_list
        if layer == 2:
            self.t3=h_list

        for hash_node_list in all_participant_hashnode_list:
            for node_id in hash_node_list:
                index1 = all_participant_hashnode_list.index(hash_node_list)
                index2 = hash_node_list.index(node_id)
                if node_id in self.node_h_dict.keys():
                    self.node_h_dict[node_id].append(h_list[index1][index2])
                else:
                    self.node_h_dict[node_id]=[h_list[index1][index2]]

        self.all_node_list=sorted(list(self.node_h_dict.keys()))


        for key in self.node_h_dict:
            h_stack=torch.stack(self.node_h_dict[key])
            self.node_h_dict[key]=torch.max(h_stack,dim=0).values
            index = self.all_node_list.index(key)
            if layer==0:
                self.m1[index]=self.node_h_dict[key]
                self.node_h_dict[key]=self.linear1(self.node_h_dict[key])
                self.node_h_dict[key]=torch.tanh(self.node_h_dict[key])
                
            if layer==1:
                self.m2[index] = self.node_h_dict[key]
                self.node_h_dict[key]=self.linear2(self.node_h_dict[key])
            if layer==2:
                self.m3[index] = self.node_h_dict[key]
                self.node_h_dict[key]=self.linear3(self.node_h_dict[key])


        
        if layer == 0:
            self.final_hidden_o1=self.linear1(self.m1)
            self.final_hidden_h1=torch.tanh(self.final_hidden_o1)
            

        if layer == 1:
            self.final_hidden_h2 = self.linear2(self.m2)

        if layer == 2:
            
            self.final_hidden_h3=torch.zeros_like(self.m3)
            
            self.final_hidden_h3 = self.linear3(self.m3)

        return self.node_h_dict

    def allocate_h_to_p(self,all_node_h):
        client_h_dict={}
        for hash_node_list in self.all_participant_hashnode_list:
            index1 = self.all_participant_hashnode_list.index(hash_node_list)
            for node_id in hash_node_list:
                index2=hash_node_list.index(node_id)
                if index2!=0:
                    client_h_dict[index1].append(all_node_h[node_id])
                else:
                    client_h_dict[index1]=[all_node_h[node_id]]
        return client_h_dict

    def cal_global_l_h3(self,grad_l_h3_list,file,norm,pos_weight):
        
        
        self.cp_global_l_h3=cp.zeros((len(self.all_node_list),self.hidden_sizes[3]))
        
        for node_id in self.all_node_list:
            index1 =self.all_node_list.index(node_id)
            for part_node_id in self.all_participant_hashnode_list:
                if node_id in part_node_id:
                    index2=part_node_id.index(node_id)
                    index3=self.all_participant_hashnode_list.index(part_node_id)
                    item=cp.asarray(grad_l_h3_list[index3][index2,:].cpu().numpy())
                    self.cp_global_l_h3[index1,:]=self.cp_global_l_h3[index1,:]+item


    def cal_l_w_jacrev(self, layer):

        
        if layer==2:
            self.cp_m3=cp.asarray(self.m3.data.cpu().numpy())
            self.cp_l_w3=cp.zeros((self.linear3.weight.shape[0],self.linear3.weight.shape[1]))
            for i in range(self.cp_global_l_h3.shape[0]):
                for j in range(self.cp_global_l_h3.shape[1]):
                    selfmade_jacob=cp.zeros((self.linear3.weight.shape[0],self.linear3.weight.shape[1]))
                    
                    selfmade_jacob[j,:]=self.cp_m3[i,:].reshape(self.cp_m3.shape[1])
                    l_h3_i_j=cp.ones_like(selfmade_jacob)
                    l_h3_i_j=l_h3_i_j*self.cp_global_l_h3[i][j]
                    self.cp_l_w3=cp.add(self.cp_l_w3,cp.multiply(l_h3_i_j,selfmade_jacob))
            
        if layer==1:
            self.cp_m2 = cp.asarray(self.m2.data.cpu().numpy())
            self.cp_l_w2 = cp.zeros((self.linear2.weight.shape[0], self.linear2.weight.shape[1]))
            for i in range(self.cp_l_h2.shape[0]):
                for j in range(self.cp_l_h2.shape[1]):
                    selfmade_jacob = cp.zeros((self.linear2.weight.shape[0], self.linear2.weight.shape[1]))
                    selfmade_jacob[j, :] = self.cp_m2[i, :].reshape(self.cp_m2.shape[1])  
                    l_h2_i_j = cp.ones_like(selfmade_jacob)
                    l_h2_i_j = l_h2_i_j * self.cp_l_h2[i][j]
                    self.cp_l_w2 = cp.add(self.cp_l_w2, cp.multiply(l_h2_i_j, selfmade_jacob))
        if layer==0:
            self.cp_m1 = cp.asarray(self.m1.data.cpu().numpy())
            self.cp_l_w1 = cp.zeros((self.linear1.weight.shape[0], self.linear1.weight.shape[1]))
            self.cp_o1=cp.asarray(self.final_hidden_o1.data.cpu().numpy())
            h1_o1 = cp.ones((self.cp_o1.shape[0],self.cp_o1.shape[1]))-cp.power(cp.tanh(self.cp_o1),2)
            
            self.cp_l_o1=cp.multiply(self.cp_l_h1,h1_o1)
            for i in range(self.cp_l_h1.shape[0]):
                for j in range(self.cp_l_h1.shape[1]):
                    selfmade_jacob = cp.zeros((self.linear1.weight.shape[0], self.linear1.weight.shape[1]))
                    selfmade_jacob[j, :] = self.cp_m1[i, :].reshape(self.cp_m1.shape[1])  
                    l_o1_i_j = cp.ones_like(selfmade_jacob)
                    l_o1_i_j = l_o1_i_j * self.cp_l_o1[i][j]
                    self.cp_l_w1 = cp.add(self.cp_l_w1, cp.multiply(l_o1_i_j, selfmade_jacob))


    def cal_l_m_jacrev(self, layer):

        if layer==1:
            self.cp_l_m3=cp.zeros_like(self.cp_m3)
            for i in range(self.final_hidden_h3.shape[0]):
                for j in range(self.final_hidden_h3.shape[1]):
                    selfmade_jacob=cp.zeros((self.cp_m3.shape[0],self.cp_m3.shape[1]))
                    cp_item=cp.asarray(self.linear3.weight[j,:].reshape(self.linear3.weight.shape[1]).data.cpu().numpy())
                    selfmade_jacob[i,:]=cp_item

                    l_h3_i_j=cp.ones_like(selfmade_jacob)
                    l_h3_i_j=l_h3_i_j*self.cp_global_l_h3[i][j]
                    self.cp_l_m3= cp.add(self.cp_l_m3, cp.multiply(l_h3_i_j, selfmade_jacob))
        if layer==0:
            self.cp_l_m2 = cp.zeros_like(self.cp_m2)
            for i in range(self.final_hidden_h2.shape[0]):
                for j in range(self.final_hidden_h2.shape[1]):
                    selfmade_jacob = cp.zeros((self.cp_m2.shape[0], self.cp_m2.shape[1]))
                    cp_item = cp.asarray(
                        self.linear2.weight[j, :].reshape(self.linear2.weight.shape[1]).data.cpu().numpy())
                    selfmade_jacob[i, :] = cp_item
                    l_h2_i_j = cp.ones_like(selfmade_jacob)
                    l_h2_i_j = l_h2_i_j * self.cp_l_h2[i][j]
                    self.cp_l_m2 = cp.add(self.cp_l_m2, cp.multiply(l_h2_i_j, selfmade_jacob))
    def cal_l_t(self, layer):
        if layer==1:
            self.cp_l_t3 = cp.zeros_like(self.cp_m3)
            for i in range(self.m3.shape[0]):
                for j in range(self.m3.shape[1]):

                    m3_t3_jacobian=cp.zeros_like(self.cp_l_t3)
                    m3_t3_jacobian[i][j]=1
                    l_m3_i_j=cp.ones_like(m3_t3_jacobian)
                    l_m3_i_j=l_m3_i_j*self.cp_l_m3[i][j]
                    self.cp_l_t3=cp.add(self.cp_l_t3,cp.multiply(l_m3_i_j,m3_t3_jacobian))
            
        if layer==0:
            self.cp_l_t2 = cp.zeros_like(self.cp_m2)
            for i in range(self.m2.shape[0]):
                for j in range(self.m2.shape[1]):
                    m2_t2_jacobian=cp.zeros_like(self.cp_l_t2)
                    m2_t2_jacobian[i][j]=1
                    l_m2_i_j=cp.ones_like(m2_t2_jacobian)
                    l_m2_i_j=l_m2_i_j*self.cp_l_m2[i][j]
                    self.cp_l_t2=cp.add(self.cp_l_t2,cp.multiply(l_m2_i_j,m2_t2_jacobian))


    def cal_l_h(self, layer, part_list):
        if layer==1:
            self.cp_l_h2 = cp.zeros((len(self.all_node_list),self.hidden_sizes[2]))
            for i in range(self.m3.shape[0]):
                for j in range(self.m3.shape[1]):
                    t3_h2_jacobian=cp.zeros_like(self.cp_l_h2)
                    client_id_list = []
                    tensor_list = []
                    for client in range(len(part_list)):

                        if self.all_node_list[i] in part_list[client].hash_node_list:
                            client_id_list.append(client)
                            client_global_index = part_list[client].hash_node_list.index(self.all_node_list[i])
                            tensor_list.append(part_list[client].ah_dict2[client_global_index].values[j])
                    max_line=torch.max(torch.stack(tensor_list,dim=0),dim=0)
                    
                    max_client=client_id_list[max_line.indices.item()]
                    src_nodeclient_global_index = part_list[max_client].hash_node_list.index(self.all_node_list[i])
                    inner_index=part_list[max_client].ah_dict2[src_nodeclient_global_index].indices[j]
                    dst_node_client_global_index=part_list[max_client].edges_dict[src_nodeclient_global_index][inner_index]
                    dst_node_cor_index=self.all_node_list.index(part_list[max_client].hash_node_list[dst_node_client_global_index])
                    grad_value=part_list[max_client].dgl_graph.ndata['norm'][src_nodeclient_global_index]*part_list[max_client].dgl_graph.ndata['norm'][dst_node_client_global_index]
                    t3_h2_jacobian[dst_node_cor_index][j]=grad_value.item()

                    l_t3_i_j = cp.ones_like(t3_h2_jacobian)
                    l_t3_i_j = l_t3_i_j * self.cp_l_t3[i][j]
                    self.cp_l_h2 = cp.add(self.cp_l_h2, cp.multiply(l_t3_i_j, t3_h2_jacobian))
            

        if layer==0:
            self.cp_l_h1 = cp.zeros((len(self.all_node_list), self.hidden_sizes[1]))
            for i in range(self.m2.shape[0]):
                for j in range(self.m2.shape[1]):
                    t2_h1_jacobian = cp.zeros_like(self.cp_l_h1)
                    client_id_list = []
                    tensor_list = []
                    for client in range(len(part_list)):
                        if self.all_node_list[i] in part_list[client].hash_node_list:
                            client_id_list.append(client)
                            client_global_index = part_list[client].hash_node_list.index(self.all_node_list[i])
                            tensor_list.append(part_list[client].ah_dict1[client_global_index].values[j])
                    max_line = torch.max(torch.stack(tensor_list, dim=0), dim=0)

                    max_client = client_id_list[max_line.indices.item()]
                    src_nodeclient_global_index = part_list[max_client].hash_node_list.index(self.all_node_list[i])
                    inner_index = part_list[max_client].ah_dict1[src_nodeclient_global_index].indices[j]
                    dst_node_client_global_index = part_list[max_client].edges_dict[src_nodeclient_global_index][
                        inner_index]
                    dst_node_cor_index = self.all_node_list.index(
                        part_list[max_client].hash_node_list[dst_node_client_global_index])
                    grad_value = part_list[max_client].dgl_graph.ndata['norm'][src_nodeclient_global_index] * \
                                 part_list[max_client].dgl_graph.ndata['norm'][dst_node_client_global_index]
                    t2_h1_jacobian[dst_node_cor_index][j] = grad_value.item()

                    l_t2_i_j = cp.ones_like(self.cp_l_h1)
                    l_t2_i_j = l_t2_i_j * self.cp_l_t2[i][j]
                    self.cp_l_h1 = cp.add(self.cp_l_h1, cp.multiply(l_t2_i_j,  t2_h1_jacobian))

    def perform_kmeans(self, edge_path, comms_path, file):
        path = '../data/real'

        nmi1, ari1, eq1 = adapt_dgl_hash__kmeans_onmi(edge_path, self.final_hidden_h3, comms_path, path,
                                                 file)

        return nmi1, ari1, eq1
