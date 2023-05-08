import time

import dgl
import torch
import networkx as nx
from functorch import jacrev
from torch.nn import Parameter

from optimizer import loss_function_AE
from layers import GCNConv, ClientGCNConv
from tool import hash
import cupy as cp
from utils import preprocess_graph
import scipy.sparse as sp
import torch.nn.functional as F


class Participant:

    def __init__(self, edge_path, feat_path, args, hidden_sizes):
        self.g = nx.read_edgelist(edge_path, nodetype=int)  
        
        
        self.feat = self.load_feat_data(self.g, feat_path) 
        self.dgl_graph=dgl.from_networkx(self.g, node_attrs=['feat'])
        self.adj_label=self.dgl_graph.adj().to_dense()
        self.adj_label=self.adj_label.cpu()+torch.eye(len(self.dgl_graph.nodes())).cpu()
        self.feat = self.dgl_graph.ndata['feat'].double()
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0] = self.feat.shape[1]
        self.clientmodel1 = ClientGCNConv().cuda()
        self.clientmodel2 = ClientGCNConv().cuda()
        self.clientmodel3 = ClientGCNConv().cuda()

    def load_feat_data(self, g, feat_path):
        torch.set_printoptions(profile='full')  
        feat = []
        with open(feat_path, 'r') as f:
            for node in f:
                node = node.strip('\n').split(' ')
                node = [eval(x) for x in node]
                
                try:
                    g.nodes[node[0]]['feat'] = node[1:]
                except:
                    pass

        for i in g.nodes:
            
            feat.append(g.nodes[i]['feat'])

        feat_tensor = torch.DoubleTensor(feat).cuda()
        return feat_tensor

    def generate_hash_dict(self):

        hash_dict = dict()
        for node in self.g.nodes():
            hash_dict[hash(node)] = node
        self.hash_dict = hash_dict

        self.hash_node_list=sorted(list(self.g.nodes()))
        return hash_dict



    def process_g(self):
        features =self.dgl_graph.ndata['feat']
        
        self.dgl_graph = dgl.add_self_loop(self.dgl_graph)
        
        deg = self.dgl_graph.in_degrees().double()
        norm = torch.pow(deg, -0.5)
        norm[torch.isinf(norm)] = 0
        self.dgl_graph.ndata['norm'] = norm.unsqueeze(1)
        self.dgl_graph.ndata['f'] = features

    def save_t(self, h, layer):
        if layer == 0:
            self.t1=h
        if layer == 1:
            self.t2=h
        if layer == 2:
            self.t3=h


    def update_h(self,h_list,layer):
        index=0

        if layer==0:
            self.hidden_vector1=torch.tensor([item.cpu().detach().numpy() for item in h_list]).cuda()
        if layer==1:
            self.hidden_vector2=torch.tensor([item.cpu().detach().numpy() for item in h_list]).cuda()
            
        if layer==2:
            self.hidden_vector3=torch.tensor([item.cpu().detach().numpy() for item in h_list]).cuda()

    def get_global_num(self, global_edges, global_nodes):
        
        
        self.pos_weight = torch.DoubleTensor([(global_nodes * global_nodes - global_edges) / global_edges]).cuda()
        self.norm = torch.DoubleTensor([global_nodes * global_nodes / (global_nodes * global_nodes - global_edges * 2)]).cuda()
        

    def cal_loss_h3(self):
        self.hidden_vector3.requires_grad_()
        recover_matrix=torch.mm(self.hidden_vector3,self.hidden_vector3.T)

        
        
        loss = loss_function_AE(preds=recover_matrix.cuda(), labels=self.adj_label.cuda(),
                                norm=self.norm, pos_weight=self.pos_weight)
        grad_l_h3=torch.autograd.grad(loss, self.hidden_vector3)[0]
        return loss,grad_l_h3

    def cal_ah_for_tn_hn(self, layer):
        
        self.edges_dict={}
        for index in range(self.dgl_graph.edges()[0].shape[0]):
            src=self.dgl_graph.edges()[0][index]
            dst=self.dgl_graph.edges()[1][index]
            
            if src.item() not in self.edges_dict.keys():
                self.edges_dict[src.item()]=[dst.item()]
            else:
                self.edges_dict[src.item()].append(dst.item())

        if layer==1:
            
            self.ah_dict2={}
            for node in self.edges_dict.keys():
                for neighbour in self.edges_dict[node]:
                    
                    
                    h2=self.hidden_vector2[neighbour]
                    ah2=h2*self.dgl_graph.ndata['norm'][node]*self.dgl_graph.ndata['norm'][neighbour]
                    if node not in self.ah_dict2.keys():
                        self.ah_dict2[node]=[ah2]
                    else:
                        self.ah_dict2[node].append(ah2)
                self.ah_dict2[node]=torch.max(torch.stack(self.ah_dict2[node],dim=0),dim=0)
                
            
        if layer==0:
            
            self.ah_dict1 = {}
            for node in self.edges_dict.keys():
                for neighbour in self.edges_dict[node]:
                    
                    
                    h1 = self.hidden_vector1[neighbour]
                    ah1 = h1 * self.dgl_graph.ndata['norm'][node] * self.dgl_graph.ndata['norm'][neighbour]
                    if node not in self.ah_dict1.keys():
                        self.ah_dict1[node] = [ah1]
                    else:
                        self.ah_dict1[node].append(ah1)
                self.ah_dict1[node] = torch.max(torch.stack(self.ah_dict1[node], dim=0), dim=0)
                
