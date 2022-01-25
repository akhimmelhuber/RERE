
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from random import choice
from explainer.utils import k_hop_subgraph, deleteNode, deleteEdge, gnnx_result
from torch_geometric.utils import remove_isolated_nodes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




class PolicyExplainerEnv(object):
    """
    Description:
        A subgraph around the target node with k-hop nodes from the target node is used as input
        as well as the corresponding labels and features of the nodes.
        The edges are removed sequentially in order to arrive at a reduced subgraph that explains the
        decision making process of the classification algorithm.
    """

    def __init__(self, node_idx_list, edge_index_list, features_list, entropy_class, num_hops, hops_policy, prog_args):
    
        
        self.node_idx_list = node_idx_list
        node_idx = choice(self.node_idx_list)
        self.args = prog_args
        self.node_idx = node_idx
        self.edge_index_list = edge_index_list
        self.entropy_class = entropy_class
        self.graph_mode = prog_args.graph_mode
     
        if self.graph_mode:
             idx = self.node_idx_list.index(node_idx)
             self.edge_index_subset = self.edge_index_list
             self.edge_index = self.edge_index_subset
             self.features = features_list

        else: 

           nodes_subset, edge_index_subset, edge_mask= k_hop_subgraph(node_idx, num_hops, edge_index_list)
           self.edge_index = edge_index_list
           self.num_hops = num_hops
           self.hops_policy = hops_policy
       
           self.edge_index_subset = edge_index_subset
         
           self.features = features_list

        

    def numberOfInputs(self, edge_index_subset):
        return edge_index_subgraph.shape

    def numberOfOutputs(self, exclude_node):
        possibleActions = [node for node in self.possibleActions if node not in exclude_node]
        size = len(possibleActions)
        return size

    def actionsDict(self, possibleActions):
        Dict = dict(zip(possibleActions, range(0, len(possibleActions))))
        return Dict

    def step(self, state, action, node_idx, stop_action_sampled):
        done = False
        if not stop_action_sampled:
            
            resultingState = deleteEdge(state, action)
            
            
            dictr = self.actionsDict(np.unique(resultingState.cpu()))
            edge_index_graph = torch.tensor(
                [[dictr[x] for x in resultingState.tolist()[0]], [dictr[x] for x in resultingState.tolist()[1]]])
            edge_index_graph = edge_index_graph.long()
            data = Data(edge_index=edge_index_graph, num_nodes=len(np.unique(resultingState.cpu())))
            G = to_networkx(data, to_undirected=True)

            number_comp = nx.number_connected_components(G)
            isolates = list(nx.isolates(G))

 
            
            if self.graph_mode:
                if (len(np.unique(resultingState.cpu())) == 0):
                    done = False
                    return state, done 
                
                else:
            
                    num_nodes = data.num_nodes
                    out = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes)
                    data.edge_index, data.edge_attr, mask = out
                    
                    G = to_networkx(data, to_undirected=True)
                    largest_cc = max(nx.connected_components(G), key=len)
                    #print("largest cc", largest_cc)
                    G_subgraph = G.subgraph(largest_cc).copy() 
                    
                    result = [e for e in G_subgraph.edges]
                    for t in range(len(result)):
                           result.append((result[t][1], result[t][0]))
    
                  
                    start_0 = []
                    start_1 = []
                    
                    for j in range(len(result)):
                       start_0.append(result[j][0])
                       start_1.append(result[j][1])
                       
                 
                    resultingState = torch.stack([torch.LongTensor(start_0), torch.LongTensor(start_1)])        
                    label_switch, prob_before, prob_after, label_new, label_org = self.entropy_class.switch(node_idx, self.features, self.edge_index, resultingState) 

                    if (label_org == label_new) :
    
                         done = bool(action == -1)
                         return resultingState,  done
    
                    else:
                         done = False
                         return state, done
 
                    return resultingState, done
            
            
            else:
               if (len(np.unique(resultingState.cpu())) == 0) or (node_idx not in (np.unique(resultingState.cpu()))):

                done = False
                return state, done

               else:
                    label_switch, prob_before, prob_after, label_new, label_org = self.entropy_class.switch(node_idx, self.features, self.edge_index, resultingState) 
                    if (label_org == label_new) :
    
                         done = bool(action == -1)
                         return resultingState,  done
    
                    else:
                         print("label switch constraint")
                         done = False
                         return state, done

                    return resultingState, done
                    
        else:
            done = True
            return state, done

    def reward(self, resultingState, state, node_idx):

        reward = 0 
        return reward  # reward

    def reward_final(self, resultingState, state, node_idx):
        org, diff, final = self.entropy_class.logits(node_idx, self.features, self.edge_index, resultingState)
        reward = diff.item()

        return reward

    def reset(self, edge_index):
        #node_idx = self.node_idx_list[2]  
        node_idx = choice(self.node_idx_list)
        
        if self.graph_mode:
             idx = self.node_idx_list.index(node_idx)
             edge_index_subset = self.edge_index[idx] 
             self.node_idx = idx
        
        else:
        
               nodes_subset, edge_index_subset, edge_mask = k_hop_subgraph(node_idx, self.num_hops, edge_index[0],
                                                                    relabel_nodes=False,
                                                                    num_nodes=None, flow='source_to_target')
        
               self.node_idx = node_idx
        self.state = edge_index_subset
        self.steps_beyond_done = None
        return self.state, self.node_idx

