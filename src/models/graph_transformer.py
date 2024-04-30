import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_mincut_pool

from .vit import *
from .gcn import GCN



class Classifier(nn.Module):
    """
    Implementation of Graph Transformer adapted for multi-class classification.
    """
    def __init__(self, n_class):
        super(Classifier, self).__init__()
        self.d_in = 512 # patch size
        self.n_feature = 64
        self.n_layer = 3
        self.n_node_cluster = 100

        self.gc = GCN(self.d_in, self.n_feature)                          # 64 -> 128
        self.pool = nn.Linear(self.n_feature, self.n_node_cluster)        # 100 -> 20
        self.transformer = ViT(self.n_feature, n_class)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.n_feature))
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x, labels, adj_matrix, mask, graphcam_flag=False):   
        x = mask.unsqueeze(2) * x
        x = self.gc(x, adj_matrix, mask)
        s = self.pool(x)

        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            torch.save(s_matrix, 'graphcam/s_matrix.pt')
            torch.save(s[0], 'graphcam/s_matrix_ori.pt')
            
            if path.exists('graphcam/att_1.pt'):
                os.remove('graphcam/att_1.pt')
                os.remove('graphcam/att_2.pt')
                os.remove('graphcam/att_3.pt')
    
        x, adj_matrix, mc1, o1 = dense_mincut_pool(x, adj_matrix, s, mask)
        b, _, _ = x.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        out = self.transformer(x)

        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1

        pred = out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out)
            torch.save(p, 'graphcam/prob.pt')
            index = np.argmax(out.cpu().data.numpy(), axis=-1)

            for index_ in range(3):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(x.device), 
                                               method="transformer_attribution", 
                                               is_ablation=False,
                                               start_layer=0, 
                                               **kwargs
                                               )

                torch.save(cam, 'graphcam/cam_{}.pt'.format(index_))

        return pred, labels, loss