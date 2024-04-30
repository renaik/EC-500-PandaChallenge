#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os

torch.backends.cudnn.deterministic = True



def collate(batch):
    """
    b[0] = 'feature'
    b[1] = 'label'
    b[2] = 'id'
    b[3] = 'adj_s'
    """
    feature = [ b[0] for b in batch ] # w, h
    label = [ b[1] for b in batch ]
    id = [ b[2] for b in batch ]
    adj_s = [ b[3] for b in batch ]

    return {'feature': feature, 'label': label, 'id': id, 'adj_s': adj_s}



def preparefeatureLabel(batch_graph, batch_label, batch_adjs):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    masks = torch.zeros(batch_size, max_node_num)
    adjs =  torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feats = torch.zeros(batch_size, max_node_num, 512)

    for i in range(batch_size):
        cur_node_num =  batch_graph[i].shape[0]
        # Node attribute feature.
        tmp_node_fea = batch_graph[i]
        batch_node_feats[i, 0:cur_node_num] = tmp_node_fea

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        
        # masks
        masks[i,0:cur_node_num] = 1  

    node_feats = batch_node_feats.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feats, labels, adjs, masks



def train(sample, model):
    node_feats, labels, adjs, masks = preparefeatureLabel(sample['feature'],
                                                          sample['label'],
                                                          sample['adj_s'])
    labels, preds, loss = model.forward(node_feats, labels, adjs, masks)

    return labels, preds, loss



def evaluate(sample, model, graphcam_flag=False):
    node_feats, labels, adjs, masks = preparefeatureLabel(sample['feature'],
                                                          sample['label'],
                                                          sample['adj_s'])
    if not graphcam_flag:
        with torch.no_grad():
            labels, preds, loss = model.forward(node_feats, labels, adjs, masks)
    else:
        torch.set_grad_enabled(True)
        labels, preds, loss = model.forward(node_feats, labels, adjs, masks, graphcam_flag=graphcam_flag)
        
    return labels, preds, loss