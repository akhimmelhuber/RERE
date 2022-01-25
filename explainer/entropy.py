from explainer.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
import torch
from torch.distributions import Categorical
import numpy as np

def T_scaling(logits, temperature):
  return torch.div(logits, temperature)  
  

class Entropy(object):
    def __init__(self, net, graph_mode, log=True):
        super(Entropy, self).__init__()
        self.net = net
        self.log = log
        self.graph_mode = graph_mode

    def __flow__(self):
        for module in self.net.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = edge_index.size(0), edge_index.size(1)

        subset, edge_index, edge_mask = k_hop_subgraph(
            node_idx, 2, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())


        x = x[subset]
        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, edge_mask, subset, kwargs

    def __entropy__(self, p):

        return Categorical(probs=p).entropy()

    def switch(self, node_idx, x_org, edge_index_org, edge_index_2, **kwargs):

        self.net.eval()
        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, edge_mask, subset, kwargs = self.__subgraph__(
            node_idx, x_org, edge_index_org, **kwargs)
            
        adj = to_dense_adj(edge_index).cpu()
        x_reshaped = x.unsqueeze(0).cpu()

        ypred, adj_att = self.net(x_reshaped, adj)
        

        if self.graph_mode:
            node_pred = ypred
            log_logits = nn.Softmax(dim=1)(node_pred)[0]
            
            pred_label = log_logits.argmax(dim=-1).item()

            
        else:

            node_pred = ypred[0, 0, :]
            log_logits = nn.Softmax(dim=0)(node_pred)
            pred_label = log_logits.argmax(dim=-1).item()
        

        x_2, edge_index_2, edge_mask, subset, kwargs = self.__subgraph__(
            node_idx, x_org, edge_index_2, **kwargs)

        adj_2 = to_dense_adj(edge_index_2).cpu()
        x_2_reshaped = x_2.unsqueeze(0).cpu()

        ypred_final, adj_att = self.net(x_2_reshaped, adj_2)
        
        if self.graph_mode:
            node_pred_final = ypred_final
            log_logits_final = nn.Softmax(dim=1)(node_pred_final)[0]
            pred_label_final = log_logits_final.argmax(dim=-1).item()

        else:
           
           node_pred_final = ypred_final[0, 0, :]
           log_logits_final = nn.Softmax(dim=0)(node_pred_final)
           pred_label_final = log_logits_final.argmax(dim=-1).item()

        
        prob_before = log_logits[pred_label]
        prob_after = log_logits_final[pred_label]

        label_before = pred_label
        label_after = pred_label_final

        if label_before == label_after:
            label_switch = 0
        else:
            label_switch = 1

        return label_switch, prob_before, prob_after, label_after, label_before

    def logits(self, node_idx, x_org, edge_index_org, edge_index_final, **kwargs):

        self.net.eval()
        
        if self.graph_mode:
             edge_index = edge_index_org
             subset = np.unique(edge_index.cpu()[0])
             x = x_org[subset]
           
        else: 
             # Only operate on a k-hop subgraph around `node_idx`.
             x, edge_index, edge_mask, subset, kwargs = self.__subgraph__(
                 node_idx, x_org, edge_index_org, **kwargs)

        adj = to_dense_adj(edge_index).cpu()
        x_reshaped = x.unsqueeze(0).cpu()
        ypred, adj_att = self.net(x_reshaped, adj)
        
        if self.graph_mode:
            node_pred = ypred
            log_logits = nn.Softmax(dim=1)(node_pred)[0]            
            pred_label = log_logits.argmax(dim=-1).item()

            
        else:

            node_pred = ypred[0, 0, :]
            log_logits = nn.Softmax(dim=0)(node_pred)
            pred_label = log_logits.argmax(dim=-1).item()

        entropy_org = self.__entropy__(log_logits)
        
        if self.graph_mode:
             
             s_0 = edge_index_final
             num_nodes = s_0.max().item() + 1

             subset = np.unique(s_0.cpu()[0])

             row, col = s_0
             relabelled = row.new_full((num_nodes, ), -1)
             relabelled[subset] = torch.arange(len(subset), device=row.device)
             edge_index_2 = relabelled[s_0]
             x_2 = x_org[subset]  
        
        else: 
             x_2, edge_index_2, edge_mask, subset, kwargs = self.__subgraph__(
                 node_idx, x_org, edge_index_final, **kwargs)

        adj_2 = to_dense_adj(edge_index_2).cpu()
        x_2_reshaped = x_2.unsqueeze(0).cpu()

        ypred_final, adj_att = self.net(x_2_reshaped, adj_2)
        
        if self.graph_mode:
            node_pred_final = ypred_final
            log_logits_final = nn.Softmax(dim=1)(node_pred_final)[0]
            pred_label_final = log_logits_final.argmax(dim=-1).item()

        else:
           node_pred_final = ypred_final[0, 0, :]
           log_logits_final = nn.Softmax(dim=0)(node_pred_final)
           pred_label_final = log_logits_final.argmax(dim=-1).item()

        entropy_final = self.__entropy__(log_logits_final)

        diff = entropy_org - entropy_final

        return entropy_org, diff, entropy_final
