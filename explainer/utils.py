import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import models


def deleteNode(edge_list, node_id):
    """
    Removes all entries of node_id from edge_list
    """
    inds = np.where(edge_list == node_id)
    mask = torch.ones(edge_list.shape[1])
    print(inds[1])
    mask[inds[1]] = 0
    tensor = edge_list[:, mask.to(torch.bool)]

    return tensor
    

def deleteEdge(edge_list, node_id):
    """
    Removes all entries of node_id from edge_list
    """
    ind_rev = np.where((edge_list[1] == edge_list[0][node_id]) & (edge_list[0] == edge_list[1][node_id]))
    inds = np.array([node_id, ind_rev[0][0]])
    mask = torch.ones(edge_list.shape[1])
    mask[inds] = 0
    tensor = edge_list[:, mask.to(torch.bool)]

    return tensor
    
    
def gnnx_result(node_id, prog_args, unconstrained=False):

    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]    # get computation graph

    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]
 
    model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.load_state_dict(ckpt["model_state"]) 
    writer = None
  

    experiment = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=False,
        graph_idx=prog_args.graph_idx,
    )

    result_gnnx = experiment.explain(node_id, prog_args, unconstrained=False)
    return result_gnnx


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    #print("num nodes", num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
        
    #print("row", row)

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    #print("subsets", subsets)

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    subset = torch.cat(subsets).unique()
  
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
               
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)       
        edge_index = node_idx[edge_index]


    return subset, edge_index, edge_mask


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def discount_rewards(rewards, gamma):
    t_steps = np.arange(len(rewards))
    r = np.array(rewards) * gamma ** t_steps
    r = r[::-1].cumsum()[::-1] / gamma ** t_steps
    return r


def subgraph(node_idx, x, edge_index, hops_policy):
    subset, edge_index, edge_mask = k_hop_subgraph(
        node_idx, hops_policy, edge_index, relabel_nodes=True,
        num_nodes=None, flow='source_to_target')

    x = x[subset]

    return x, edge_index, edge_mask, subset


def numberOfOutputs(node_idx, edge_index, exclude_node, num_hops):
    nodes_subset, edge_index_subset, edge_mask = k_hop_subgraph(node_idx, num_hops, edge_index)
    nodes_minus = torch.cat([nodes_subset[0:0], nodes_subset[0 + 1:]]).tolist()
    nodes_minus.append(-1)
    possibleActions = [node for node in nodes_minus if node not in exclude_node]
    size = len(possibleActions)
    return size


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)


def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = adj[graph_idx][neighbors][:, neighbors]
        sub_feat = feat[graph_idx, neighbors]
        sub_label = label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors


