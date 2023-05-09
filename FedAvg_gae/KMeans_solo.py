import pickle as pkl
from sklearn.cluster import KMeans
from collections import defaultdict
import sys
import os
import numpy as np
import networkx as nx
import numpy as np
import math
import copy
from sklearn import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def NMI(A,B):
    A= np.array(A)
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

def calc_jaccard_score(set1, set2):
    intsect_nodes = set1.intersection(set2)
    union_nodes = set1.union(set2)
    jaccard_score = len(intsect_nodes) / len(union_nodes)
    return jaccard_score

def kmeans_onmi(edge_path, hidden_emb,comms_path, dataset_name):

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
    pre_path='../pre_result/'

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

    
    nmi=NMI(com, real_com)
    ari = metrics.adjusted_rand_score(real_com, com)
    eq = calc_EQ(edge_path, pred_path)

    return nmi,ari,eq

def get_real_com(comms_path,edge_path):
    g = nx.read_edgelist(edge_path, nodetype=int)
    my_seq = list(g.nodes())
    nodes = len(my_seq)
    real_path = comms_path
    real_com = [0 for x in range(0, nodes)]
    with open(real_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            if len(arr) == 1:
                arr = arr[0]
                arr = line.split('\t')
            for a in arr:
                try:
                    real_com[int(my_seq.index(int(a)))] = cnt
                except:
                    pass
            cnt += 1

        real_com = np.array(real_com)

    return real_com,cnt,nodes,my_seq

def get_comms_solo(edge_path, hidden_emb, dataset_name,cnt):

    data = hidden_emb
    g = nx.read_edgelist(edge_path, nodetype=int)
    my_seq = list(g.nodes())
    nodes = len(my_seq)
    com = [0 for x in range(0, nodes)]
    n_clusters = cnt - 1

    data = data.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, verbose=0, random_state=7) #set same paramerter
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
    pre_path = '../pre_result/'

    with open(pre_path + 'pre_' + dataset_name + '.txt', 'w') as f:
        f.write(s)

    pred_path = pre_path + 'pre_' + dataset_name + '.txt'


    with open(pred_path, 'r') as f:
        cnt = 1
        for line in f.readlines():
            line = line.strip('\n').strip()
            arr = line.split(' ')
            for a in arr:
                com[int(my_seq.index(int(a)))] = cnt
            cnt += 1
        com = np.array(com)

    pre_coms=dict(zip(my_seq, com))

    new_dict = {}
    pre_dict=[]
    for k, v in pre_coms.items():
        new_dict.setdefault(v, []).append(k)
    for i in new_dict.keys():
        pre_dict.append(set(new_dict[i]))

    return com,pre_dict

def comb_comms(edge_path,pred_comms_list,real_com,my_seq,filename,nodes):


        com = [0 for x in range(0, nodes)]
        comb_comms = list()
        
        copy_pred_comms_list = copy.deepcopy(pred_comms_list)

        temp_list=[]
        temp_remove_list=[]
        for i in range(len(pred_comms_list)):
            for j in range(len(pred_comms_list[i])):
                temp_list.append(copy.deepcopy(pred_comms_list[i][j]))
        temp_remove_list=copy.deepcopy(temp_list[len(pred_comms_list[0]):])
        max_similarity_score=0.01
        for i in range(len(pred_comms_list[0])):
            for temp_com in temp_remove_list:
                if len(temp_list[i].intersection(temp_com)) > 0:
                    temp_similarity_score = calc_jaccard_score(temp_list[i],temp_com)
                    if temp_similarity_score>max_similarity_score:
                        temp_list[i].union(temp_com)
                        temp_remove_list.remove(temp_com)

        for i in range(len(pred_comms_list[0])):
            comb_comms.append(sorted(temp_list[i]))
        pred_comms = [set(t) for t in set(tuple(_) for _ in comb_comms) if len(t) > 0]


        # for i in range(len(pred_comms_list)):
        #     for j in range(len(pred_comms_list[i])):  
        #         for m in range(len(copy_pred_comms_list)):
        #             comm = list()
        #             max_similarity_score = 0.001
        #             if i != m:
        #                 for n in range(len(copy_pred_comms_list[m])): 
        #                     if len(pred_comms_list[i][j].intersection(copy_pred_comms_list[m][n])) > 0:
        #                         temp_similarity_score = calc_jaccard_score(pred_comms_list[i][j],
        #                                                                         copy_pred_comms_list[m][n])

        #                         if temp_similarity_score > max_similarity_score:
        #                             comm = copy_pred_comms_list[m][n]
        #                             max_similarity_score = temp_similarity_score
        #             pred_comms_list[i][j] = pred_comms_list[i][j].union(comm) 
        # for pred_comms in pred_comms_list:
        #     for pred_comm in pred_comms:
        #         comb_comms.append(sorted(pred_comm))
        # pred_comms = [set(t) for t in set(tuple(_) for _ in comb_comms) if len(t) > 0]  


        for i in range(len(pred_comms)):
            for j in pred_comms[i]:
                com[my_seq.index(j)]=i+1
        pred_path='../pre_result/'+'pre_' + filename + '.txt'

        nmi = metrics.normalized_mutual_info_score(com, real_com)
        ari = metrics.adjusted_rand_score(real_com, com)

        eq = calc_EQ( edge_path,pred_path)

        return nmi, ari, eq

