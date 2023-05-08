import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def load_data(edge_path,feat_path):
    
    G = nx.Graph()
    edge_list = []
    with open(edge_path, 'r') as f:
        for edge in f:
            edge = edge.strip('\n').split(' ')
            if len(edge)==1:
                edge=edge[0]
                edge = edge.strip('\n').split('\t')
            edge = [eval(x) for x in edge]
            edge_list.append((edge[0], edge[1]))

    G.add_edges_from(edge_list)
    adj = nx.adjacency_matrix(G)
    feat = []
    with open(feat_path, 'r') as f:
        for node in f:
            node = node.strip('\n').split(' ')
            node = [eval(x) for x in node]
            
            try:
                G.nodes[node[0]]['feat'] = node[1:]
            except:
                pass

    for i in G.nodes:
        feat.append(G.nodes[i]['feat'])

    feat_tensor=torch.DoubleTensor(feat)

    return adj, feat_tensor

def single_load_data(edge_path,feat_path):
    
    G = nx.Graph()
    edge_list = []
    with open(edge_path, 'r') as f:
        for edge in f:
            edge = edge.strip('\n').split(' ')
            if len(edge)==1:
                edge=edge[0]
                edge = edge.strip('\n').split('\t')
            edge = [eval(x) for x in edge]
            edge_list.append((edge[0], edge[1]))

    G.add_edges_from(edge_list)
    adj = nx.adjacency_matrix(G)
    feat = []
    with open(feat_path, 'r') as f:
        for node in f:
            node = node.strip('\n').split(' ')
            node = [eval(x) for x in node]
            
            try:
                G.nodes[node[0]]['feat'] = node[1:]
            except:
                pass

    for i in G.nodes:
        feat.append(G.nodes[i]['feat'])

    feat_tensor=torch.DoubleTensor(feat)

    return adj, feat_tensor,G.nodes

def single_preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def preprocess_graph(g,adj,overlapping_degree):
    
    
    
    node_list=list(g.nodes())
    
    node_degree={}
    for node_id in overlapping_degree.keys():
        node_degree[node_id]=node_list.index(node_id)
    
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1)) 
    for node_id in overlapping_degree.keys():
        
            
            
        rowsum[node_degree[node_id]]=overlapping_degree[node_id]+1
        
    
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float64)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)