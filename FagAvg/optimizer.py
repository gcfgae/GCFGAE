import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def loss_function_AE(preds, labels, norm, pos_weight):

    norm=torch.DoubleTensor([norm])
    pos_weight=torch.DoubleTensor([pos_weight])


    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
   
    return cost

def loss_function_MAE(preds, labels, norm, pos_weight):

    norm=torch.DoubleTensor([norm])
    pos_weight=torch.DoubleTensor([pos_weight])
    loss2 = torch.DoubleTensor(preds.shape)

    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

    cost = loss_fn(preds, labels)
    cost=cost.sum()
  

    return cost



def loss_function_X(preds,labels,norm):
    cost=norm*F.binary_cross_entropy_with_logits(preds, labels)
    return cost


