import torch
import torch.nn as nn
def gcn_message_func(edges):
    w = edges.src['norm'] * edges.dst['norm']
    
    return {'h': edges.src['f'] * w}
    

def gcn_reduce_func(nodes):

    return {'s': torch.max(nodes.mailbox['h'], 1).values}

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats,act):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats,bias=False)
        self.act =act


    def forward(self, g, f):
        g.ndata['f'] = f
        g.update_all(gcn_message_func, gcn_reduce_func)
        s=g.ndata['s']
        g.ndata['f'] = self.linear(g.ndata['s'])
        g.ndata['f'] = self.act(g.ndata['f'])
        f = g.ndata.pop('f')
        return f,s

class ClientGCNConv(nn.Module):
    def __init__(self):
        super(ClientGCNConv, self).__init__()

    def forward(self, g, f):
        g.ndata['f'] = f
        g.update_all(gcn_message_func, gcn_reduce_func)
        g.ndata['f'] = g.ndata['s'] 
        f = g.ndata.pop('f')
        return f