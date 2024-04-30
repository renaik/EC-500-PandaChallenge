import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """ Graph Convolutional Network Layer """

    def __init__(self, d_in, d_out, add_self=True, bias=True, normalize=True, bn=True, dropout=0.):
        super(GCN,self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.add_self = add_self
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out).cuda())
        else:
            self.bias = None
        self.normalize = normalize
        self.bn = bn
        if bn:
            self.bn_layer = torch.nn.BatchNorm1d(d_out)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out).cuda())
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x, adj_matrix, mask):
        y = torch.matmul(adj_matrix, x)

        if self.add_self:
            y += x

        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias

        if self.normalize:
            y = F.normalize(y, p=2, dim=2)
            
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index),y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index=[]
            ssum = 0

            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)

            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)

            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]] = bn_tensor_bf[start_index[i]:start_index[i+1]]
            y = bn_tensor_af

        if self.dropout > 0.0:
            y = self.dropout_layer(y)

        y = F.relu(y)
        
        return y