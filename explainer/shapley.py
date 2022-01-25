"""
Code adapted from:
https://github.com/divelab/DIG/tree/main

A Turnkey Library for Diving into Graph Deep Learning Research, 2021
Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Zhao Xu and Haiyang Yu and Jingtun Zhang and Yi Liu and Keqiang Yan and Bora Oztekin and Haoran, Liu and Xuan Zhang and Cong Fu and Shuiwang Ji
"""


import copy
import torch
import numpy as np
from typing import Callable, Union
from scipy.special import comb
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_adj

def GnnNetsGC2valueFunc(gnnNets, target_class):
    def value_func(batch):
        with torch.no_grad():
            logits = gnnNets(data=batch)
            probs = F.softmax(logits, dim=-1)
            score = probs[:, target_class]
        return score
    return value_func


def GnnNetsNC2valueFunc(gnnNets_NC, node_idx: Union[int, torch.Tensor], target_class: torch.Tensor):
    def value_func(x_r, edge_index_r):
        with torch.no_grad():
            
            logits, adj_att = gnnNets_NC(x_r, edge_index_r)

            probs = F.softmax(logits, dim=-1)
            batch_size = 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score
    return value_func
    
def GnnNetsNC2valueFuncGraph(gnnNets_NC, node_idx: Union[int, torch.Tensor], target_class: torch.Tensor):
    def value_func(x_r, edge_index_r):
        with torch.no_grad():
          
            logits, adj_att = gnnNets_NC(x_r, edge_index_r)

            probs = F.softmax(logits, dim=-1)
            batch_size = 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])

            score = probs[0][0][target_class]

            return score
    return value_func
    
    
def GnnNetsNC2valueFunclabel(gnnNets_NC, node_idx: Union[int, torch.Tensor], target_class: torch.Tensor):
    def value_func(x_r, edge_index_r):
        with torch.no_grad():
            logits, adj_att = gnnNets_NC(x_r, edge_index_r)

            probs = F.softmax(logits, dim=-1)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx]
            score = score.argmax().item()

            return score
    return value_func
    
def GnnNetsNC2valueFunclabel_graph(gnnNets_NC, node_idx: Union[int, torch.Tensor], target_class: torch.Tensor):
    def value_func(x_r, edge_index_r):
        with torch.no_grad():
            logits, adj_att = gnnNets_NC(x_r, edge_index_r)

            probs = F.softmax(logits, dim=-1)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs
            score = score.argmax().item()
            return score
            
    return value_func


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError



def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_x = X * node_mask.unsqueeze(1)
    return ret_x, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def gnn_score(coalition: list, x, edge_index, value_func: Callable,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = len(np.unique(edge_index.cpu()[0]))  
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(x.device)
    mask[coalition] = 1.0

    ret_x, ret_edge_index = subgraph_build_func(x, edge_index, mask)
    
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    ret_x = ret_x.unsqueeze(0).type(torch.FloatTensor).to(x.device)
    ret_edge_index = to_dense_adj(ret_edge_index)
    score = value_func(ret_x, ret_edge_index)
    return score.item()

def new_label(coalition: list, x, edge_index, value_func: Callable,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = len(np.unique(edge_index.cpu()[0]))   
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(x.device)
    mask[coalition] = 1.0

    ret_x, ret_edge_index = subgraph_build_func(x, edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])

    ret_x = ret_x.unsqueeze(0).type(torch.FloatTensor).to(x.device)
    ret_edge_index = to_dense_adj(ret_edge_index)
    score = value_func(ret_x, ret_edge_index)

    return score


