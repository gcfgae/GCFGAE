import datetime
import pandas as pd
import torch
import sys
from torch import optim
from KMeans_solo import kmeans_onmi,comb_comms,get_comms_solo,get_real_com
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from singleGAE import SingleClient
from server import Server
from optimizer import loss_function_AE
import numpy as np
import argparse
from copy import deepcopy,copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--real', type=str, default=True, help='type of dataset.')
    parser.add_argument('--file', type=str, default='citeseer', help='type of dataset.')
    parser.add_argument('--epsilon', type=float, default=0.8, help='epsilon of dp.')

    return parser.parse_args()

if __name__ == '__main__':
    file_name_list = []
    isreal = True
    global_epoch_list = []
    local_epoch_list = []
    for filename in file_name_list:
        for epoch in global_epoch_list: 
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.set_printoptions(profile="full", precision=4)
            torch.set_default_tensor_type(torch.DoubleTensor)
            args = get_args()

            args.file = filename
            args.global_epochs = epoch
            path = './data/real/'
            nparts_list = [2,4,6,8,10]

            for local_epoch in local_epoch_list:
                for nparts in nparts_list:
                    
                    client_list = [SingleClient(args, edge_path_list[i], feat_path_list[i], hidden_sizes) for i in range(nparts)]

                    server = Server()
                    server.K=len(client_list)

                    for c in client_list:
                        aj,fe,node_list=c.load_data()
                        c.hidden_sizes[0]=fe.shape[1]
                        c.prepare_data()
                        c.init_weight(model_dict)
                        c.optimizer = optim.Adam(c.single_model.parameters(), lr=args.local_lr)
                        c.adj_norm = c.adj_norm.to_dense()

                    for g_epoch in range(epoch): 
                        local_weight_gc2 ={}
                        local_weight_gc3 = {}
                        for c in client_list:
                          
                            c.train_model(local_epoch)
                            torch.set_printoptions(profile='full')

                            gc1_weight=c.single_model.gc1.weight
                            gc2_weight=c.single_model.gc2.weight
                            gc3_weight=c.single_model.gc3.weight
                            local_weight_gc1[client_list.index(c)]=deepcopy(c.single_model.gc1.weight)
                          
                            local_weight_gc2[client_list.index(c)] = deepcopy(c.single_model.gc2.weight)
                            local_weight_gc3[client_list.index(c)] = deepcopy(c.single_model.gc3.weight)

                           
                        global_wight_gc1, global_wight_gc2, global_wight_gc3=server.aggregate_weight(local_weight_gc1,local_weight_gc2,local_weight_gc3)
     
                        for c in client_list:
                            c.update_weight(global_wight_gc1, global_wight_gc2, global_wight_gc3)
                        
                    nmi_list = []
                    ari_list = []
                    eq_list = []
                    coms_list = []
                    pre_coms_list = []
                    nodes = 0
                    real_coms, cnt, nodes, my_seq = get_real_com(comms_path, edge_path)
                    for c in client_list:
                        coms, pre_coms = deepcopy(get_comms_solo(c.edge_path, c.hidden_vextor, args.file, cnt))
                        coms_list.append(deepcopy(coms))
                        pre_coms_list.append(deepcopy(pre_coms))
                    nmi, ari, eq = comb_comms(edge_path, pre_coms_list, real_coms, my_seq, args.file, nodes)
                   



