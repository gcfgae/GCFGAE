from collections import Counter

import networkx as nx
import scipy.sparse as sp

import itertools
from itertools import combinations_with_replacement
import numpy as np
import torch
import random
import math
import time
from copy import deepcopy
from torch.nn import functional as F
from multiprocessing.pool import Pool

class Server:
    def __init__(self):
        self.norm=0
        self.C=0.5
        self.K=0
        torch.set_default_tensor_type(torch.DoubleTensor)

    def aggregate_weight(self,local_weight_gc1,local_weight_gc2,local_weight_gc3):
        if self.K==2:
            m=2
        else:
            m=np.max([int(self.C * self.K), 1])
        index = random.sample(range(0, self.K), m)
        global_wight_gc1 = deepcopy(local_weight_gc1[index[0]])
        global_wight_gc2 = deepcopy(local_weight_gc2[index[0]])
        global_wight_gc3 = deepcopy(local_weight_gc3[index[0]])
        for num in range(1,len(index)):
            global_wight_gc1=global_wight_gc1+local_weight_gc1[index[num]]
            global_wight_gc2=global_wight_gc2+local_weight_gc2[index[num]]
            global_wight_gc3=global_wight_gc3+local_weight_gc3[index[num]]
        global_wight_gc1=global_wight_gc1/len(index)
        global_wight_gc2=global_wight_gc2/len(index)
        global_wight_gc3=global_wight_gc3/len(index)
        return global_wight_gc1,global_wight_gc2,global_wight_gc3




