import time

import torch
import networkx as nx
from functorch import jacrev
from torch.nn import Parameter
from tool import hash
import cupy as cp
from utils import preprocess_graph
import scipy.sparse as sp
from model import  OP_Net
import torch.nn.functional as F


class Participant:

    def __init__(self, edge_path, feat_path, args, hidden_sizes,all_nodes,ans_fp):
        self.g = nx.read_edgelist(edge_path, nodetype=int)  
        self.adj = nx.adjacency_matrix(self.g)
        self.edges = torch.DoubleTensor(self.adj.toarray()).sum().cuda()
        self.feat = self.load_feat_data(self.g, feat_path)
        self.overlapping_node_dict = {}  
        self.overlapping_nodes = set()  
        self.non_overlapping_node_num = 0  
        self.local_overlapping_node_dict = {}  
        self.client_num = -1
        self.encrypt_overlapping_degree = {}  
        self.nodeid_line_dict = {}  
        self.global_loss = 0
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0] = self.feat.shape[1]
        
        self.clientmodel1 = OP_Net(self.hidden_sizes[0], self.hidden_sizes[1], args, lambda x: x).cuda()
        self.clientmodel2 = OP_Net(self.hidden_sizes[1], self.hidden_sizes[2], args, lambda x: x).cuda()
        self.clientmodel3 = OP_Net(self.hidden_sizes[2], self.hidden_sizes[3], args, lambda x: x).cuda()
        self.hash_dict=self.generate_hash_dict()
        self.local_l_h3=cp.zeros((all_nodes,hidden_sizes[3]),dtype='float64')
        self.ss_total_time=0
        self.ans_fp=ans_fp


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

    def load_degree_randommatrix(self, random_matrix):
        self.random_matrix = random_matrix

    def append_overlapping_nodes(self, sub_overlapping_nodes, client):

        self.overlapping_nodes = self.overlapping_nodes.union(sub_overlapping_nodes) 


        for node_id in sub_overlapping_nodes:
            
            self.overlapping_node_dict.setdefault(self.hash_dict[node_id], []).append(client)
            
        

    def generate_hash_dict(self):

        hash_dict = dict()
        for node in self.g.nodes():
            hash_dict[hash(node)] = node
        self.hash_dict = hash_dict
        return hash_dict

    def get_overlapping_degree(self):
        self.degrees = {}
        self.overlapping_degree = {}  
        for (node, val) in self.g.degree():
            self.degrees.setdefault(node, val)
        
        
        temp = self.degrees.keys() & (self.overlapping_node_dict.keys())
        for term in temp:
            self.overlapping_degree[term] = self.degrees[term]
        

    def encrypt_degree(self):
        self.get_overlapping_degree()
        
        for node in self.overlapping_degree.keys():
            for num in self.overlapping_node_dict[node]:
                self.overlapping_degree[node] = self.overlapping_degree[node] + self.random_matrix[num]
                
                
            self.encrypt_overlapping_degree[hash(node)] = self.overlapping_degree[node]  
            
        
        return self.encrypt_overlapping_degree

    def load_degree(self, update_overlapping_degrees):
        
        
        for node in update_overlapping_degrees.keys():
            
            unhash_node = self.hash_dict[node]
            
            self.overlapping_degree[unhash_node] = update_overlapping_degrees[node]


    def prepare_data(self):
        
        
        self.adj_norm = preprocess_graph(self.g, self.adj, self.overlapping_degree).cuda()
        
        self.adj_label = self.adj + sp.eye(self.adj.shape[0])
        self.adj_label = torch.DoubleTensor(self.adj_label.toarray()).cuda()
        self.pos_weight = (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

        self.norm =self.adj.shape[0] * self.adj.shape[0] / (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum() * 2)
        self.grad_adj = self.adj_norm.to_dense()
        self.adj_norm2 = self.adj_norm.clone().to_dense()

        

    def get_parameter(self, model_dict):
        self.clientmodel1.gc.weight = Parameter(model_dict['gc1.weight'].cuda())
        self.cupy_gc1_weight = cp.asarray(model_dict['gc1.weight'].numpy())
        self.clientmodel2.gc.weight = Parameter(model_dict['gc2.weight'].cuda())
        self.cupy_gc2_weight = cp.asarray(model_dict['gc2.weight'].numpy())
        self.clientmodel3.gc.weight = Parameter(model_dict['gc3.weight'].cuda())
        self.cupy_gc3_weight = cp.asarray(model_dict['gc3.weight'].numpy())

    def update_cp_weight(self):
        self.cupy_gc1_weight = cp.asarray(self.clientmodel1.gc.weight.cpu().numpy())
        self.cupy_gc2_weight = cp.asarray(self.clientmodel2.gc.weight.cpu().numpy())
        self.cupy_gc3_weight = cp.asarray(self.clientmodel3.gc.weight.cpu().numpy())

    def get_nodeid_line_dict(self):
        self.nodeid_line_dict = {}
        for node in self.g.nodes:
            self.nodeid_line_dict[hash(node)] = list(self.g.nodes).index(node)

    def get_h_overlap_dict(self, hidden_vector):
        h_overlap_dict = {}

        for node in self.overlapping_nodes:
            h_overlap_dict[node] = self.secert_share_enc(hidden_vector[self.nodeid_line_dict[node]],self.random_matrix,self.ans_fp,self.overlapping_node_dict[self.hash_dict[node]])

        self.h_overlap_dict = h_overlap_dict

        return h_overlap_dict, list(self.overlapping_nodes)



    def get_T_tensor_new(self, support):
        list_nnk = []
        for n1 in self.overlapping_nodes:
            n1_value = self.nodeid_line_dict[n1]  
            list_A = []

            support_clone = self.support_tensor.clone()
            tensor_q = support_clone[n1_value][:]
            tensor_q = tensor_q.reshape(1, support_clone.shape[1])

            list_A.append(self.adj_norm[n1_value][n1_value])
            tensor_A = torch.Tensor(list_A).reshape(len(list_A), 1)
            ans=torch.mul(tensor_A, tensor_q)
            list_nnk.append(self.secert_share_enc(ans,self.random_matrix,self.ans_fp,self.overlapping_node_dict[self.hash_dict[n1]]))
            

        if len(list_nnk)!=0:
            self.tensor_T = torch.stack(list_nnk, 0)
            flag=1
        else:
            flag=0
            self.tensor_T=list_nnk
        return flag,self.tensor_T

    def save_support_hidden_vector_output(self, support, hidden_vector, layer, grad_adj, output):
        self.support_tensor = support
        self.grad_adj = grad_adj
        self.cp_grad_adj= cp.asarray(grad_adj.cpu().numpy(),dtype='float64')

        self.hidden_vector = hidden_vector
        
        if layer == 0:
            self.support_tensor1 = support
            self.hidden_vector1 = hidden_vector
            self.output1 =output
            self.cp_output1 =cp.asarray(output.cpu().numpy())

        if layer == 1:
            self.support_tensor2 = support
            self.hidden_vector2 = hidden_vector
            self.output2 = output
            self.cp_output2 = cp.asarray(output.cpu().numpy())

        if layer == 2:
            self.support_tensor3 = support
            self.hidden_vector3 = hidden_vector
            self.output3 = output
            self.cp_output3 = cp.asarray(output.cpu().numpy())

    def change_h(self, final_h_dict, layer):

        for node in final_h_dict.keys():

            real_node = node
            index = self.nodeid_line_dict[real_node]
            if layer == 0:
                self.hidden_vector1.data[index] = final_h_dict[node].data  
                self.hidden_vector.data[index] = final_h_dict[node].data  

            if layer == 1:
                self.hidden_vector2.data[index] = final_h_dict[node].data  
                self.hidden_vector.data[index] = final_h_dict[node].data  
                
            if layer == 2:
                self.hidden_vector3.data[index] = final_h_dict[node].data  
                self.hidden_vector.data[index] = final_h_dict[node].data  
                
        if layer == 0:
            
            self.hidden_vector1 = F.tanh(self.output1)
            self.cp_hidden_vector1=cp.asarray(self.hidden_vector1.cpu().numpy())
            self.cp_output1=cp.asarray(self.output1.cpu().numpy())
        if layer == 1:
            self.cp_hidden_vector2 = cp.asarray(self.hidden_vector2.cpu().numpy())
        if layer == 2:
            self.cp_hidden_vector3 = cp.asarray(self.hidden_vector3.cpu().numpy())


            
        del final_h_dict


    def get_non_overlapping(self):
        
        
        self.non_overlapping_node_num = len(set(self.nodeid_line_dict.keys()) - self.overlapping_nodes) 
        

    def get_global_num(self, global_edges, global_nodes):
        self.pos_weight = torch.DoubleTensor([(global_nodes * global_nodes - global_edges) / global_edges]).cuda()
        self.norm = torch.DoubleTensor([global_nodes * global_nodes / (global_nodes * global_nodes - global_edges * 2)]).cuda()
        

    def get_l_h(self, layer):
        if layer == 0:
            return self.cp_local_l_h1
        if layer == 1:
            return self.cp_local_l_h2
        if layer == 2:
            return self.cp_local_l_h3

    def get_overlap_node_list(self):
        return self.overlapping_nodes

    def update_grad_adj(self, make_to_zero_list):
        for node in make_to_zero_list:
            
            index = self.nodeid_line_dict[node]
            self.grad_adj[index][index] = 0
        self.cp_grad_adj=cp.asarray(self.grad_adj.cpu().numpy())

    def a_x_w_3(self, x):
        support = torch.mm(x, self.clientmodel3.gc.weight.data)  
        output = torch.spmm(self.adj_norm2,
                            support)  
        self.adj_norm2.data = self.grad_adj.data
        del support, x

        return output

    def single_a_x_w_3(self, x,i,j):
        
        
        
        output = torch.spmm(self.adj_norm2.data[i, :].reshape(1, self.adj_norm2.shape[1]),
                            torch.mm(x, (self.clientmodel3.gc.weight.data[:,j]).reshape(self.clientmodel3.gc.weight.shape[0],1)))  
        
        del  x

        return output

    def a_x_w_2(self, x):
        support = torch.mm(x, self.clientmodel2.gc.weight.data)  
        output = torch.spmm(self.adj_norm2,
                            support)  
        self.adj_norm2.data = self.grad_adj.data
        del support, x
        return output

    def jacrev_jacob(self, layer1, layer2):
        with torch.no_grad():
            if layer1 == 3:
                
                l_h2 =cp.zeros((self.hidden_vector2.shape[0],self.hidden_vector2.shape[1]),dtype='float64')
                for i in range(self.hidden_vector3.shape[0]):
                    for j in range(self.hidden_vector3.shape[1]):

                        A=self.cp_grad_adj[i, :].reshape(1, self.cp_grad_adj.shape[1])
                        
                        W=(self.cupy_gc3_weight[:,j]).reshape(self.cupy_gc3_weight.shape[0],1)
                        jacob_ans=cp.matmul(A.T,W.T)

                        tmp = self.cp_local_l_h3[i][j]*jacob_ans
                        l_h2=cp.add(l_h2,tmp)
                        del jacob_ans, tmp
                        torch.cuda.empty_cache()

                
                self.cp_local_l_h2 = l_h2
                
            if layer1 == 2:

                
                l_h2 =cp.zeros((self.hidden_vector1.shape[0],self.hidden_vector1.shape[1]),dtype='float64')
                for i in range(self.hidden_vector2.shape[0]):
                    for j in range(self.hidden_vector2.shape[1]):

                        A = self.cp_grad_adj[i, :].reshape(1, self.cp_grad_adj.shape[1])
                        W = (self.cupy_gc2_weight[:, j]).reshape(self.cupy_gc2_weight.shape[0], 1)
                        jacob_ans = cp.matmul(A.T, W.T)

                        tmp = self.cp_local_l_h2[i][j] * jacob_ans
                        l_h2 = cp.add(l_h2, tmp)

                        del jacob_ans, tmp
                        torch.cuda.empty_cache()

                self.cp_local_l_h1 = l_h2

        return l_h2


    def generate_overlap_node_grad(self, l_h):
        node_grad_dict = {}
        for op_node in self.overlapping_nodes:
            node_grad_dict[op_node] = self.cp_secert_share_enc(l_h[self.nodeid_line_dict[op_node]],self.random_matrix,self.ans_fp,self.overlapping_node_dict[self.hash_dict[op_node]])
            
        return node_grad_dict

    def cp_update_h_grad(self, layer, new_op_grad):
        if layer == 2:
            for op_node in new_op_grad:
                self.cp_local_l_h2[self.nodeid_line_dict[op_node]] = new_op_grad[op_node]

        if layer == 1:
            for op_node in new_op_grad:
                self.cp_local_l_h1[self.nodeid_line_dict[op_node]] = new_op_grad[op_node]

    def cp_get_h_w(self, layer):
        if layer == 3:
            tensor_h2_w3 = torch.mm(self.grad_adj, self.hidden_vector2).T
            h2_w3 = cp.matmul(self.cp_grad_adj, self.cp_hidden_vector2).T
            self.local_l_w3_grad = cp.matmul(h2_w3, self.cp_local_l_h3)

            
            return self.cp_secert_share_enc(self.local_l_w3_grad,self.random_matrix,self.ans_fp)
        if layer == 2:
            h2_w2 = cp.matmul(self.cp_grad_adj, self.cp_hidden_vector1).T
            
            self.local_l_w2_grad = cp.matmul(h2_w2, self.cp_local_l_h2)
            
            return self.cp_secert_share_enc(self.local_l_w2_grad,self.random_matrix,self.ans_fp)
        if layer == 1:
            tensor_o1_w1 = torch.mm(self.grad_adj, self.feat).T
            self.cp_feat=cp.asarray(self.feat.cpu().numpy())
            o1_w1 = cp.matmul(self.cp_grad_adj, self.cp_feat).T


            tensor_h1_o1 = torch.pow(torch.tanh(self.output1), 2)
            h1_o1 = cp.ones((self.cp_output1.shape[0],self.cp_output1.shape[1]))-cp.power(cp.tanh(self.cp_output1),2)

            

            l_o1 = cp.multiply(self.cp_local_l_h1, h1_o1)
            
            self.local_l_w1_grad = cp.matmul(o1_w1, l_o1)

            
            return self.cp_secert_share_enc(self.local_l_w1_grad,self.random_matrix,self.ans_fp)

    def get_local_overlapping_node_dict(self):
        for node_1 in list(self.overlapping_nodes):
            i = list(self.overlapping_nodes).index(node_1)
            
            self.local_overlapping_node_dict[node_1] = i

    def cal_local_loss(self,recover_matrix):
        local_loss = F.binary_cross_entropy_with_logits(recover_matrix, self.adj_label, pos_weight=self.pos_weight)
        return local_loss
    def calculate_local_lA(self):
        
        
        

        cp_recover_matrix = cp.matmul(self.cp_hidden_vector3, self.cp_hidden_vector3.T)
        
        
        
        reala = cp.asarray(self.adj_label.cpu().numpy())
        cp_pos_weight=cp.asarray(self.pos_weight.cpu().numpy())
        cp_one=cp.ones((cp_recover_matrix.shape[0],cp_recover_matrix.shape[1]))

        self.cp_local_l_A= -(cp_pos_weight * reala * ((cp.exp(-cp_recover_matrix)) / (cp_one + cp.exp(-cp_recover_matrix)))) + (
                (cp_one - reala) / ((cp_one + cp.exp(-cp_recover_matrix))))

    def secert_share_enc(self,raw,ramdom_number,ans_fp,only_op=None):
        enc_start_time=time.time()
        if only_op!= None:
            random_list=[]
            for op_partcipant in only_op:
                random_list.append(ramdom_number[op_partcipant])
            ramdom_number=random_list
        for number in ramdom_number:
            raw=torch.add(raw,number)
        enc_end_time = time.time()
        ans_f=open(ans_fp,'a+')
        self.ss_total_time=self.ss_total_time+enc_end_time-enc_start_time
        
        return raw
    def cp_secert_share_enc(self,raw,ramdom_number,ans_fp,only_op=None):
        enc_start_time = time.time()
        if only_op!= None:
            random_list=[]
            for op_partcipant in only_op:
                random_list.append(ramdom_number[op_partcipant])
            ramdom_number=random_list
        for number in ramdom_number:
            raw=cp.add(raw,number)
        enc_end_time = time.time()
        
        
        self.ss_total_time = self.ss_total_time + enc_end_time - enc_start_time
        return raw


    def cal_local_l_h3_2(self,node1,node2,jacob_ans_1):
        self.local_l_h3=self.local_l_h3+cp.multiply(self.cp_local_l_A[self.nodeid_line_dict[node1]][
                        self.nodeid_line_dict[node2]], jacob_ans_1) * 2
    def cal_local_l_h3(self,node1,node2,jacob_ans_1):
        self.local_l_h3=self.local_l_h3+cp.multiply(self.cp_local_l_A[self.nodeid_line_dict[node1]][
                        self.nodeid_line_dict[node2]], jacob_ans_1)