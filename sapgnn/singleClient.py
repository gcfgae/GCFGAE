import dgl
import torch

from utils import single_load_data, single_preprocess_graph
from model import SingleGAE
import scipy.sparse as sp

class SingleClient:
    def __init__(self, args, edge_path, feat_path, hidden_sizes):
        self.edge_path = edge_path
        self.feat_path = feat_path
        
        self.dgl_graph,self.adj, self.feature,self.nodes_list= self.load_data()
        
        
        self.adj_label = self.dgl_graph.adj().to_dense()
        self.adj_label = self.adj_label + torch.eye(len(self.dgl_graph.nodes()))

        self.feature = self.dgl_graph.ndata['feat'].double()  
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes[0]=self.feature.shape[1] 
        self.single_model=SingleGAE(self.hidden_sizes,args).double()
    def load_data(self):
        G,adj, features,node_list = single_load_data(self.edge_path, self.feat_path)
        return G,adj, features,node_list
    def prepare_data(self):
        self.n_nodes, self.feat_dim = self.feature.shape

        self.adj_norm = single_preprocess_graph(self.adj)

        
        
        self.pos_weight = (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()
        
        self.norm = self.adj.shape[0] * self.adj.shape[0] / (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum() * 2)
        
        self.att_norm = self.feature.shape[1] * self.feature.shape[1] / (
                    self.feature.shape[1] * self.feature.shape[1] - self.feature.sum() * 2)
    def process_g(self):
        features =self.dgl_graph.ndata['feat']
        
        self.dgl_graph = dgl.add_self_loop(self.dgl_graph)
        
        deg = self.dgl_graph.in_degrees().double()
        norm = torch.pow(deg, -0.5)
        norm[torch.isinf(norm)] = 0
        self.dgl_graph.ndata['norm'] = norm.unsqueeze(1)
        self.dgl_graph.ndata['f'] = features
