import time
from collections import Counter
from itertools import combinations_with_replacement

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd.functional import jacobian
from torch.nn import functional as F
from functools import partial
from functorch import vmap, vjp, jacrev

import cupy as cp


from torch import optim

from Kmeans import kmeans_onmi, adapt_hash_kmeans_onmi


class Server:
    def __init__(self):
        self.participant_degree = {}
        self.allnodes_num = 0
        self.alledges_num = 0
        self.overlapping_nodes_num = 0  
        self.norm = 0
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    def get_encrypt_degree(self, part_enc_degree):
        n_part = len(part_enc_degree)
        temp = set()
        for i in range(n_part):
            temp = temp | set(part_enc_degree[i].keys())
            
        for hash_node in temp:
            self.participant_degree.setdefault(hash_node, [])
        for node_id in self.participant_degree.keys():
            for i in range(n_part):
                if part_enc_degree[i].get(node_id):
                    self.participant_degree[node_id].append(part_enc_degree[i].get(node_id))
            for j in range(n_part):
                if part_enc_degree[j].get(node_id):
                    
                    part_enc_degree[j][node_id] = sum(self.participant_degree[node_id])
                    
        
        return part_enc_degree

    def get_client_make_adj_zero_dict(self, nodeId_clientId_dict):
        
        client_make_adj_zero_dict = {}
        for node in nodeId_clientId_dict.keys():
            for client in nodeId_clientId_dict[node]:
                if (nodeId_clientId_dict[node].index(client) != 0):
                    if client in client_make_adj_zero_dict.keys():
                        client_make_adj_zero_dict[client].append(node)
                    else:
                        client_make_adj_zero_dict[client] = [node]
        return client_make_adj_zero_dict

    def get_nodeID_clientID_dict(self, all_overlap_node_dict):
        nodeID_clientID_dict = {}
        
        for key in all_overlap_node_dict.keys():
            for node in all_overlap_node_dict[key]:
                
                
                if node in nodeID_clientID_dict.keys():
                    nodeID_clientID_dict[node].append(key)
                else:
                    nodeID_clientID_dict[node] = [key]
        
        
        self.nodeID_clientID_dict = nodeID_clientID_dict
        self.clientID_nodeID_dict = all_overlap_node_dict

        return nodeID_clientID_dict

    def get_all_h_overlap_dict(self, h_overlap_dict_list):
        all_h_overlap_dict = {}
        index_client = 0
        for h_overlap_dict in h_overlap_dict_list:
            for key in h_overlap_dict.keys():
                if key in all_h_overlap_dict.keys():
                    all_h_overlap_dict[key].append(h_overlap_dict[key])  
                else:
                    all_h_overlap_dict[key] = [h_overlap_dict[key]]
            index_client = index_client + 1

        return all_h_overlap_dict

    def generate_overlap_node_h(self, all_h_overlap_dict, nodeId_clientId_dict,
                                clientID_nodeID_dict, hidden_sizes, hash_overlapnodeid_list_server,
                                part_list):
        for nodeid in all_h_overlap_dict.keys():
            clients_list = nodeId_clientId_dict[nodeid]
            nodeid_times_dict = {}  
            node_union_set = []  
            all_node_list = []
            nodeid_singleClient_dict = {}  
            for client_id in clients_list:
                node_union_set = set(clientID_nodeID_dict[client_id]).union(node_union_set)
                all_node_list = all_node_list + clientID_nodeID_dict[client_id]
                for nodeid_tmp in clientID_nodeID_dict[client_id]:
                    if nodeid_tmp not in nodeid_singleClient_dict.keys():
                        nodeid_singleClient_dict[nodeid_tmp] = client_id
            nodeid_times_dict = Counter(all_node_list)

            first = 0
            final_h = torch.zeros([1, hidden_sizes[3]])
            for h in all_h_overlap_dict[nodeid]:


                if first == 0:
                    final_h = h
                    first = 1
                else:
                    final_h = final_h + h
                    temph = final_h


                
                

            for nodeid2 in nodeid_singleClient_dict:
                client = nodeid_singleClient_dict[nodeid2]
                num = nodeid_times_dict[nodeid2] - 1
                index1 = hash_overlapnodeid_list_server[client].index(nodeid2)
                index2 = hash_overlapnodeid_list_server[client].index(nodeid)

                final_h = final_h - num * part_list[client].tensor_T[index1][index2]
                

            all_h_overlap_dict[nodeid] = final_h
        return all_h_overlap_dict

    def generate_overlap_node_h_plus(self, all_h_overlap_dict, nodeId_clientId_dict,
                                     clientID_nodeID_dict, hidden_sizes, hash_overlapnodeid_list_server,
                                     part_list):
        
        self.overlapping_nodes_num = len(nodeId_clientId_dict)
        
        for nodeid in all_h_overlap_dict.keys():
            clients_list = nodeId_clientId_dict[nodeid]
            nodeid_times_dict = {}  
            node_union_set = []  
            all_node_list = []
            nodeid_singleClient_dict = {}  
            for client_id in clients_list:
                node_union_set = set(clientID_nodeID_dict[client_id]).union(node_union_set)
                all_node_list = all_node_list + clientID_nodeID_dict[client_id]
                for nodeid_tmp in clientID_nodeID_dict[client_id]:
                    if nodeid_tmp not in nodeid_singleClient_dict.keys():
                        nodeid_singleClient_dict[nodeid_tmp] = [client_id]
                    else:
                        nodeid_singleClient_dict[nodeid_tmp].append(client_id)
            nodeid_times_dict = Counter(all_node_list)

            first = 0
            final_h = torch.zeros([1, hidden_sizes[3]])
            for h in all_h_overlap_dict[nodeid]:

                if first == 0:
                    final_h = h
                    first = 1
                else:
                    final_h = final_h + h
                    temph = final_h

            node_tensorT_dict = {}
            for nodeid2 in nodeid_singleClient_dict:
                for client in nodeid_singleClient_dict[nodeid2]:
                    index1 = hash_overlapnodeid_list_server[client].index(nodeid2)
                    index2 = hash_overlapnodeid_list_server[client].index(nodeid)
                    if list(nodeid_singleClient_dict[nodeid2]).index(client) == 0:
                        final_tensor_T = part_list[client].tensor_T[index1][index2]
                    else:
                        for index_part_tensor_t in range(final_tensor_T.shape[0]):
                            temp_tensor_t = part_list[client].tensor_T[index1][index2]

                            if (temp_tensor_t[index_part_tensor_t] == 0 and final_tensor_T[index_part_tensor_t] != 0):
                                final_tensor_T[index_part_tensor_t] = temp_tensor_t[index_part_tensor_t]
                node_tensorT_dict[nodeid2] = final_tensor_T

                num = torch.DoubleTensor([nodeid_times_dict[nodeid2] - 1]).cuda()

                final_h = final_h - num * final_tensor_T

            all_h_overlap_dict[nodeid] = final_h
        return all_h_overlap_dict

    def generate_overlap_node_h_new(self, all_h_overlap_dict, nodeId_clientId_dict,
                                    clientID_nodeID_dict, hidden_sizes, hash_overlapnodeid_list_server,
                                    part_list):
        self.overlapping_nodes_num = len(nodeId_clientId_dict)
        for nodeid in all_h_overlap_dict.keys():
            clients_list = nodeId_clientId_dict[nodeid]
            nodeid_times_dict = {}  
            node_union_set = []  
            all_node_list = []
            nodeid_singleClient_dict = {}  
            for client_id in clients_list:
                node_union_set = set(clientID_nodeID_dict[client_id]).union(node_union_set)
                all_node_list = all_node_list + clientID_nodeID_dict[client_id]
                for nodeid_tmp in clientID_nodeID_dict[client_id]:
                    if nodeid_tmp not in nodeid_singleClient_dict.keys():
                        nodeid_singleClient_dict[nodeid_tmp] = client_id
            nodeid_times_dict = Counter(all_node_list)

            first = 0
            final_h = torch.zeros([1, hidden_sizes[3]])
            for h in all_h_overlap_dict[nodeid]:

                if first == 0:
                    final_h = h
                    first = 1
                else:
                    final_h = final_h + h
                    temph = final_h

            num = nodeid_times_dict[nodeid] - 1
            clients=nodeId_clientId_dict[nodeid]
            temp=0
            for client in clients:
                index1 = hash_overlapnodeid_list_server[client].index(nodeid)
                temp=temp+part_list[client].tensor_T[index1][0]
            final_h = final_h - temp / (num + 1) * (num)
            all_h_overlap_dict[nodeid] = final_h
        return all_h_overlap_dict

    def calculate_all_edgesnum(self, local_edges):
        self.alledges_num = sum(local_edges)

    def calculate_all_nodenum(self, local_nodes):
        self.allnodes_num = sum(local_nodes) + self.overlapping_nodes_num
        
        

    def get_all_nodes_num(self, local_edges, local_nodes):
        self.calculate_all_edgesnum(local_edges)
        self.calculate_all_nodenum(local_nodes)
        return self.alledges_num, self.allnodes_num

    def calculate_global_loss(self, node_list_dict, local_hidden_layer_dict, local_loss_dict, overlapping_dict):

        torch.set_printoptions(precision=17)
        pos_weight = torch.DoubleTensor(
            [(self.allnodes_num * self.allnodes_num - self.alledges_num) / self.alledges_num])
        global_loss = torch.DoubleTensor([0.0])
        
        
        for client in local_loss_dict.keys():
            global_loss = global_loss + local_loss_dict[client]

        whole_overlapping_node = set()  
        whole_id = set()  
        
        node_belong = {}
        for i in overlapping_dict.keys():
            whole_overlapping_node = whole_overlapping_node | (set(overlapping_dict[i].keys()))
        
        for i in node_list_dict.keys():
            whole_id = whole_id | (set(node_list_dict[i]))
            
        for i in whole_id:
            
            for j in node_list_dict.keys():  
                if i in node_list_dict[j]:
                    node_belong.setdefault(i, []).append(j)
        
        for i in whole_id:
            for j in whole_id:
                if j > i:
                    break
                if i in whole_overlapping_node and j in whole_overlapping_node and len(
                        set(node_belong[i]) & set(node_belong[j])) > 1:
                    num = len(set(node_belong[i]) & set(node_belong[j]))
                    
                    
                    client_t = set(node_belong[i]) & set(node_belong[j])  
                    
                    
                    
                    pred_loss = torch.DoubleTensor([(local_hidden_layer_dict[node_belong[i][0]][
                                                         node_list_dict[node_belong[i][0]][i]] *
                                                     local_hidden_layer_dict[node_belong[j][0]][
                                                         node_list_dict[node_belong[j][0]][j]]).sum()])
                    if i == j:
                        real_loss = torch.DoubleTensor([1.0])
                        op_loss = F.binary_cross_entropy_with_logits(pred_loss, real_loss, pos_weight=pos_weight)
                        
                        global_loss = global_loss - (num - 1) * op_loss
                        
                    else:
                        real_loss = torch.DoubleTensor([0.0])
                        
                        
                        op_loss = F.binary_cross_entropy_with_logits(pred_loss, real_loss, pos_weight=pos_weight)
                        
                        global_loss = global_loss - 2 * (num - 1) * op_loss
                    
                elif len(set(node_belong[i]) & set(node_belong[j])) < 1:
                    
                    
                    
                    pred_loss = torch.DoubleTensor([(local_hidden_layer_dict[node_belong[i][0]][
                                                         node_list_dict[node_belong[i][0]][i]] *
                                                     local_hidden_layer_dict[node_belong[j][0]][
                                                         node_list_dict[node_belong[j][0]][j]]).sum()])
                    real_loss = torch.DoubleTensor([0.0])
                    empty_loss = F.binary_cross_entropy_with_logits(pred_loss, real_loss, pos_weight=pos_weight)
                    global_loss = global_loss + 2 * empty_loss
        self.norm = torch.DoubleTensor(
            [self.allnodes_num * self.allnodes_num / (self.allnodes_num * self.allnodes_num - self.alledges_num * 2)])
        
        global_loss = self.norm * (global_loss / (self.allnodes_num * self.allnodes_num))
        
        
        return global_loss

    def get_all_op_node_grad(self, all_op_node_grad_dict_list):
        all_op_node_grad_dict = {}
        for op_node_grad_dict in all_op_node_grad_dict_list:
            for key in op_node_grad_dict.keys():
                if key in all_op_node_grad_dict.keys():
                    all_op_node_grad_dict[key] = all_op_node_grad_dict[key] + op_node_grad_dict[key]
                else:
                    all_op_node_grad_dict[key] = op_node_grad_dict[key]
        return all_op_node_grad_dict

    def predict(self, x):
        all_jacob_ans=torch.zeros(x.shape[0]*x.shape[0],x.shape[0],x.shape[1])
        for i in range(x.shape[0]):
            if i==0:
                s_time=time.time()
            for j in range(x.shape[0]):
                
                jacob_ans = jacrev(lambda x: torch.mm(x, x.T)[i][j])(x)
                
                all_jacob_ans[i*x.shape[0]+j]=jacob_ans
            if i==0:
                e_time=time.time()
                


    def cal_global_AZ_before(self, node_list_dict, local_hidden_layer_dict, overlapping_dict):

        self.overlapping_dict = overlapping_dict
        self.whole_id = set()  
        self.node_belong = {}
        for i in node_list_dict.keys():
            self.whole_id = self.whole_id | (set(node_list_dict[i]))
        self.whole_id = list(self.whole_id)
        
        
        for i in self.whole_id:
            for j in node_list_dict.keys():  
                if i in node_list_dict[j]:
                    self.node_belong.setdefault(i, []).append(j)
        
        
        self.final_hidden_vector = torch.randn(len(self.whole_id), local_hidden_layer_dict[0].shape[1]).double()
        j = 0
        for i in self.whole_id:
            self.final_hidden_vector[j] = local_hidden_layer_dict[self.node_belong[i][0]][
                node_list_dict[self.node_belong[i][0]][i]]
            j += 1
        
        self.cupy_final_hidden_vector=cp.asarray(self.final_hidden_vector.cpu().numpy())
        self.recover_matrix = torch.mm(self.final_hidden_vector.cuda(), self.final_hidden_vector.cuda().t())
        self.cp_recover_matrix=cp.matmul(self.cupy_final_hidden_vector, self.cupy_final_hidden_vector.T)
        self.hidden_vector_grad_list = {}  




    def global_backward_LZ(self, local_LZ_dict, node_list_dict):

        self.pos_weight = torch.DoubleTensor(
            [(self.allnodes_num * self.allnodes_num - self.alledges_num) / self.alledges_num])
        self.cp_pos_weight=cp.asarray(self.pos_weight.numpy())
        self.pos_weight =self.pos_weight.cuda()

        self.norm = torch.DoubleTensor(
            [self.allnodes_num * self.allnodes_num / (self.allnodes_num * self.allnodes_num - self.alledges_num * 2)]).cuda()
        global_LZ = torch.zeros_like(local_LZ_dict[0]).double()
        for item in local_LZ_dict.keys():
            global_LZ = global_LZ + local_LZ_dict[item]
        
        self.whole_overlapping_node = set()  
        for i in self.overlapping_dict.keys():
            self.whole_overlapping_node = self.whole_overlapping_node | (set(self.overlapping_dict[i].keys()))
        
        
        temp = np.zeros((len(self.whole_id), len(self.whole_id)))
        nn = 0
        mm = 0
        cal_overlapping = []
        empty_index = []
        empty = []
        for index, value in np.ndenumerate(temp):  
            
            if index[0] >= index[1]:
                
                if self.whole_id[index[0]] in self.whole_overlapping_node and self.whole_id[
                    index[1]] in self.whole_overlapping_node and len(
                    set(self.node_belong[self.whole_id[index[0]]]) & set(
                        self.node_belong[self.whole_id[index[1]]])) > 1:
                    
                    cal_overlapping.append(index)  
                    
                    
                    
                    nn = nn + 1
                elif len(set(self.node_belong[self.whole_id[index[0]]]) & set(
                        self.node_belong[self.whole_id[index[1]]])) < 1:  
                    empty_index.append(index)  
                    
                    pre_a = self.recover_matrix[index[0]][index[1]]
                    reala = torch.DoubleTensor([0.0])
                    op_LA = -(self.pos_weight * reala * ((torch.exp(-pre_a)) / (1 + torch.exp(-pre_a)))) + (
                            (1 - reala) / ((1 + torch.exp(-pre_a))))
                    op_LZ = op_LA * self.hidden_vector_grad_list[index[0]][index[1]]
                    
                    global_LZ = global_LZ + 2 * op_LZ
                    mm = mm + 1
        
        for i, j in cal_overlapping:
            num = len(set(self.node_belong[self.whole_id[i]]) & set(self.node_belong[self.whole_id[j]]))
            if i == j:
                
                pre_a = self.recover_matrix[i][j]
                
                reala = torch.DoubleTensor([1.0])
                op_LA = -(self.pos_weight * reala * ((torch.exp(-pre_a)) / (1 + torch.exp(-pre_a)))) + (
                        (1 - reala) / ((1 + torch.exp(-pre_a))))
                op_LZ = op_LA * self.hidden_vector_grad_list[i][j]
                global_LZ = global_LZ - (num - 1) * op_LZ
                
            else:
                
                
                reala = torch.DoubleTensor([0.0])
                pre_a = self.recover_matrix[i][j]
                op_LA = -(self.pos_weight * reala * ((torch.exp(-pre_a)) / (1 + torch.exp(-pre_a)))) + (
                        (1 - reala) / ((1 + torch.exp(-pre_a))))
                op_LZ = op_LA * self.hidden_vector_grad_list[i][j]
                global_LZ = global_LZ - 2 * (num - 1) * op_LZ
                
        global_LZ = (global_LZ / (len(self.whole_id) * len(self.whole_id))) * self.norm
        
        global_LZ_dict = {}
        for client_id in node_list_dict.keys():
            local_all_LZ = {}
            for i in node_list_dict[client_id].keys():
                
                local_all_LZ[i] = global_LZ[self.whole_id.index(i)]
            global_LZ_dict[client_id] = local_all_LZ
        

        return global_LZ_dict

    def make_global_lz_dict(self, node_list_dict,global_LZ):
        global_LZ_dict = {}
        for client_id in node_list_dict.keys():
            local_all_LZ = {}
            for i in node_list_dict[client_id].keys():
                
                local_all_LZ[i] = global_LZ[self.whole_id.index(i)]
            global_LZ_dict[client_id] = local_all_LZ
        return global_LZ_dict

    def global_backward_LZ_before(self):

        self.pos_weight = torch.DoubleTensor(
            [(self.allnodes_num * self.allnodes_num - self.alledges_num) / self.alledges_num])
        self.cp_pos_weight=cp.asarray(self.pos_weight.numpy())
        self.pos_weight=self.pos_weight.cuda()
        self.norm = torch.DoubleTensor(
            [self.allnodes_num * self.allnodes_num / (self.allnodes_num * self.allnodes_num - self.alledges_num * 2)])
        self.cp_norm=cp.asarray(self.norm)
        self.norm=self.norm.cuda()

        self.whole_overlapping_node = set()  
        for i in self.overlapping_dict.keys():
            self.whole_overlapping_node = self.whole_overlapping_node | (set(self.overlapping_dict[i].keys()))

    def perform_kmeans(self,edge_path,comms_path, file):
        path = '../data/real'

        nmi1, ari1, eq1= adapt_hash_kmeans_onmi(self.whole_id, edge_path, self.final_hidden_vector, comms_path, path, file)

        return nmi1, ari1, eq1

