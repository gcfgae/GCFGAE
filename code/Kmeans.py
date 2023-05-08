import pickle as pkl

import torch
from sklearn.cluster import KMeans
from collections import defaultdict


import sys
import os
import numpy as np
import networkx as nx
import numpy as np
import math
from tool import hash


from sklearn import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def NMI(A,B):
    
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    
    
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def cal_EQ(cover, G):
    vertex_community = defaultdict(lambda: set())
    for i, c in enumerate(cover):
        for v in c:
            vertex_community[v].add(i)
    m = 0.0
    for v, neighbors in G.edges():
        for n in neighbors:
            if v > n:
                m += 1

    total = 0.0
    for c in cover:
        for i in c:
            o_i = len(vertex_community[i])
            k_i = len(G[i])
            for j in c:
                o_j = len(vertex_community[j])
                if j not in G:
                    
                k_j = len(G[j])
                if i > j:
                    continue
                t = 0.0
                if j in G[i]:
                    t += 1.0 / (o_i * o_j)
                t -= k_i * k_j / (2 * m * o_i * o_j)
                if i == j:
                    total += t
                else:
                    total += 2 * t

    return round(total / (2 * m), 4)

def calc_EQ(edge_path, pred_path):
    comms = []
    with open(pred_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            arr = set(map(int, arr))
            comms.append(arr)

    g = nx.read_edgelist(edge_path)  
    
    for idx, comm in enumerate(comms):
        comm = [str(node) for node in comm]  
        comms[idx] = comm
    return cal_EQ(comms, g)

def adapt_hash_kmeans_onmi(whole_id,edge_path, hidden_emb,comms_path, name,dataset_name):
    
    
    real_path = comms_path

    g = nx.read_edgelist(edge_path, nodetype=int)
    my_seq=list(g.nodes())

    
    data=torch.zeros_like(hidden_emb)



    for node_id in my_seq:
        index1 = whole_id.index(hash(node_id))
        index2 = list(my_seq).index(node_id)
        data[index2] = hidden_emb[index1]

    nodes=len(my_seq)
    real_com = [0 for x in range(0, nodes)] 
    com = [0 for x in range(0, nodes)] 

    with open(real_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            if len(arr)==1:
                arr = arr[0]
                arr = line.split('\t')
            for a in arr:
                try:
                    real_com[int(my_seq.index(int(a)))] = cnt 
                except:
                    pass
            cnt += 1
            
        real_com = np.array(real_com)
    
    n_clusters=cnt-1

    data=data.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, verbose=0,random_state=7)
    kmeans.fit(data)
    label = kmeans.predict(data)

    every = {}
    for i in range(len(label)):
        if label[i] in every:
            every[label[i]].append(my_seq[i])
        else:
            every[label[i]] = [my_seq[i]]
    s = ''
    for i in every:
        v = every[i]
        v = [str(x) for x in v]
        s = s + " ".join(v)
        s = s + '\n'
    pre_path='./'+dataset_name+'/'
    os.mknod(pre_path + 'pre_'+dataset_name+'.txt')
    with open(pre_path + 'pre_'+dataset_name+'.txt', 'w') as f:
        f.write(s)
    
    pred_path = pre_path + 'pre_'+dataset_name+'.txt'
    
    

    with open(pred_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            for a in arr:
                com[int(my_seq.index(int(a)))] = cnt 
            cnt += 1
        com = np.array(com)
    
    nmi=metrics.normalized_mutual_info_score(real_com,com )

    ari = metrics.adjusted_rand_score(real_com, com)
    
    eq = calc_EQ(edge_path, pred_path)
    
    
    os.remove(pred_path)
    return nmi,ari,eq
def kmeans_onmi(edge_path, hidden_emb,comms_path, name,dataset_name):
    
    
    real_path = comms_path
    data=hidden_emb
    g = nx.read_edgelist(edge_path, nodetype=int)
    my_seq=list(g.nodes())
    
    nodes=len(my_seq)
    real_com = [0 for x in range(0, nodes)]
    com = [0 for x in range(0, nodes)]
    with open(real_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            if len(arr)==1:
                arr = arr[0]
                arr = line.split('\t')
            for a in arr:
                try:
                    real_com[int(my_seq.index(int(a)))] = cnt
                except:
                    pass
            cnt += 1
            
        real_com = np.array(real_com)

    
    n_clusters=cnt-1

    data=data.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, verbose=0,random_state=7)
    kmeans.fit(data)
    label = kmeans.predict(data)

    every = {}
    for i in range(len(label)):
        if label[i] in every:
            every[label[i]].append(my_seq[i])
        else:
            every[label[i]] = [my_seq[i]]
    s = ''
    for i in every:
        v = every[i]
        v = [str(x) for x in v]
        s = s + " ".join(v)
        s = s + '\n'
    pre_path='./centralized/'
    os.mknod(pre_path + 'pre_'+dataset_name+'.txt')
    with open(pre_path + 'pre_'+dataset_name+'.txt', 'w') as f:
        f.write(s)
    
    pred_path = pre_path + 'pre_'+dataset_name+'.txt'
    
    

    with open(pred_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            for a in arr:
                com[int(my_seq.index(int(a)))] = cnt
            cnt += 1
        com = np.array(com)
    
    nmi=metrics.normalized_mutual_info_score(real_com,com )
    
    
    ari = metrics.adjusted_rand_score(real_com, com)
    
    eq = calc_EQ(edge_path, pred_path)
    
    
    os.remove(pred_path)
    return nmi,ari,eq




