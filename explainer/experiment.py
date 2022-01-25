from utils import graph_utils
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, sort_edge_index
from explainer.utils import subgraph, extract_neighborhood
from explainer.entropy import Entropy
from explainer.environment import PolicyExplainerEnv
from explainer.models import *
from explainer.reinforce import reinforce
from matplotlib import pyplot as plt
from explainer.utils import k_hop_subgraph, discount_rewards
#from .base_explainer import ExplainerBase
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import matplotlib.colors as mcolors
import random
from .shapley import GnnNetsGC2valueFunc, GnnNetsNC2valueFunc, gnn_score, GnnNetsNC2valueFuncGraph
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, f1_score

class Explainer(object):
    def __init__(self, model, adj, feat, label, pred, train_idx, args, device, writer=None, print_training=True,
                 graph_mode=False,
                 graph_idx=False,
                 use_cuda=False):

        self.model = model
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.dataset = args.dataset
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj,
                                                                                    n_hops=self.n_hops,
                                                                                    use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.device = device
        self.hops_policy = args.hops_policy
        self.num_hops = args.num_hops
        self.print_training = print_training
        self.device = device
        self.policy_model = args.policy_model
        self.input_dim = args.input_dim
        self.policy_hidden_dim = args.policy_hidden_dim
        self.num_heads = args.num_heads
        self.number_episodes = args.number_episodes
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr = args.lr


    def explain(self, policy, graph_mode, x, edge_index, constraint, plot_graphs, node_idx,  t=0, **kwargs):
        """Explain a single node prediction
        """  

        policy_trained= eval(self.policy_model)(self.input_dim, self.policy_hidden_dim, self.num_heads).to(self.device)     
        policy_trained.load_state_dict(torch.load(policy), strict=False)
        policy_trained.eval()
        features = x
        
        if plot_graphs:
           dict_prob_list = []
           state_list = []
        
       
        net = self.model
        entropy_class = Entropy(net, graph_mode)

        env = PolicyExplainerEnv([node_idx], edge_index, features, entropy_class, self.num_hops, self.hops_policy, self.args)
        graphs = []
        times = 0
        step = 0

        while times <= t:

            if graph_mode:
                   
                   edge_index_subset = edge_index
                   s_0 = edge_index_subset
                   num_nodes = s_0.max().item() + 1

                   subset = np.unique(s_0.cpu()[0])
   
                   row, col = s_0
                   relabelled = row.new_full((num_nodes, ), -1)
                   relabelled[subset] = torch.arange(len(subset), device=row.device)
                   s_0_start = relabelled[s_0]
                   features_start = features[subset]  
                
        
            else: 
                    nodes_subset, edge_index_subset, edge_mask = k_hop_subgraph(node_idx, self.num_hops, edge_index,
                                                                        relabel_nodes=False, num_nodes=None,
                                                                        flow='source_to_target')
                                                                        
                                                
                    s_0 = edge_index_subset
                    features_start = features[nodes_subset]                                           
                    subset_0, s_0_start, edge_mask = k_hop_subgraph(node_idx, self.hops_policy, s_0, relabel_nodes=True, num_nodes=None,
                                                                  flow='source_to_target')

            dictr = dict(zip(np.unique(edge_index_subset.cpu()), range(0, len(np.unique(edge_index_subset.cpu())))))
            done = False
            
            while done == False:
            
                if graph_mode:
                
                   num_nodes = s_0.max().item() + 1
                   subset = np.unique(s_0.cpu()[0])
   
                   row, col = s_0
                   relabelled = row.new_full((num_nodes, ), -1)
                   relabelled[subset] = torch.arange(len(subset), device=row.device)
                   s_0_s = relabelled[s_0]
                   
                else:
             
                   subsetx, s_0, edge_maskx = k_hop_subgraph(node_idx, self.hops_policy, s_0, relabel_nodes=False, num_nodes=None,
                                                             flow='source_to_target')
                   subset, s_0_s, edge_mask = k_hop_subgraph(node_idx, self.hops_policy, s_0, relabel_nodes=True, num_nodes=None,
                                                             flow='source_to_target')
       
              
                
                features_s = features[subset]             
                probs = policy_trained(features_s, s_0_s, source = None)
                
                array = np.delete(np.unique(s_0.cpu()[0]),  np.where(np.unique(s_0.cpu()[0]) == node_idx))
                if graph_mode and node_idx >= 40:
                    new_array = array
                else:                
                    new_array = np.append([node_idx], array)
                dict_prob = dict(zip(new_array, np.around(probs.cpu().detach().numpy(),2)))

                if plot_graphs:
                        dict_prob_list.append(dict_prob)
                        state_list.append(s_0.tolist())
                        if graph_mode: 
                             
                            
                             s_0_probs = s_0.tolist()
                             edges = pd.DataFrame(    {
                                     "source": s_0_probs[0],
                                     "target": s_0_probs[1],

                                 }
                             )
                             G = nx.from_pandas_edgelist(
                                 edges,
                                 create_using=nx.MultiGraph(),
                             ) 
                             color = []
                             maxk = max(dict_prob, key=dict_prob.get)
                             #NH2  blue lightgrey lightgrey
                             #NO2 41 blue red red
                             for r in range(len(np.unique(s_0_probs[0]))):
                                  #print(features_s[r].tolist().index(1))
                                  #if list(G)[r] == maxk:
                                  #   color.append("orangered")
                                  if features_s[r].tolist().index(1)   == 0:
                                    color.append("darkgrey")
                                  elif features_s[r].tolist().index(1)   == 1:
                                    color.append("red")
                                  elif features_s[r].tolist().index(1)   == 2:
                                    color.append("green")
                                  elif features_s[r].tolist().index(1)  == 3:
                                    color.append("lightgrey")
                                  elif features_s[r].tolist().index(1)  == 4:
                                    color.append("blue")
                                  elif features_s[r].tolist().index(1)  == 5:
                                    color.append("green")
                                  elif features_s[r].tolist().index(1)  == 6:
                                    color.append("maroon")
                                  elif features_s[r].tolist().index(1)   == 7:
                                    color.append("yellow")
                                  elif features_s[r].tolist().index(1)  == 8:
                                    color.append("orange")
                                  elif features_s[r].tolist().index(1)   == 9:
                                    color.append("purple")
                                  elif features_s[r].tolist().index(1)   == 10:
                                    color.append("violet")
                                  elif features_s[r].tolist().index(1)   == 11:
                                    color.append("violet")
                                  elif features_s[r].tolist().index(1)   == 12:
                                    color.append("violet")
                                  elif features_s[r].tolist().index(1)   == 13:
                                    color.append("darkgreen")
                                  else:
                                    print(list(G)[r])
                                    print("color", features_s[r].tolist().index(1))
                   
      
                        else:
                            
                             s_0_probs = s_0.tolist()
                             maxk = max(dict_prob, key=dict_prob.get)
                            
                             edges = pd.DataFrame(    {
                                     "source": s_0_probs[0],
                                     "target": s_0_probs[1],
                                                                     }
                             )
      
                             G = nx.from_pandas_edgelist(
                                 edges,
                                 create_using=nx.MultiGraph(),
                             )                        
                             color = []
                             for r in range(len(np.unique(s_0_probs[0]))):
                                     
                                      if list(G)[r] == maxk:
                                          color.append("orangered")
                                         
                                      elif list(G)[r] == node_idx:
                                          color.append("cyan")
                                      else:
                                          color.append("darkgrey")
                           
                        plt.switch_backend("agg")
                        
                        
                        
                        
                        
                        
                        plt.title("Step "+str(step+1), fontsize=25)
                        nx.draw(G, with_labels = True, labels = dict_prob, node_color = color, font_size = 15, node_size=1000)
                        #nx.draw(G, with_labels = True, node_color = color, font_size = 15, node_size=1000)
                        #nx.draw(G, with_labels = False, node_color = color, node_size=1000)
                        plt.savefig("log/graphs/"+str(self.dataset)+"_"+str(node_idx)+"__"+str(step)+".png")
                        plt.close()
                        
                
                
                action = probs.argmax().item()
                if len(s_0_s[1]) <= constraint:
                    stop_action_sampled = True
                
                else: 
                     stop_action_sampled = action == features_s.shape[0] 
                     
                step = step + 1


                if not stop_action_sampled:
                    node_id = action
                    
                    
                    probs_target = policy_trained(features_s, s_0_s, source=action)
                    action_target = probs_target.argmax().item()
                   
                    indices = [i for i, x in enumerate(s_0_s[0]) if x == action]
                    filter_source = list(pd.Series(s_0_s.cpu()[1])[indices])
                    
                    values = range(0, len(filter_source))
                    map_ind = dict(zip(values, filter_source))
    
                    node_id_target = map_ind[action_target]
                    
                    for t in range(len(s_0_s[0])):
                      
                       if s_0_s[0][t]== node_id and s_0_s[1][t]== node_id_target:
                          edge_id = t
 

                else:
                       edge_id = None
                   
                s_1, done = env.step(s_0.cpu(), edge_id, node_idx, stop_action_sampled)
               

                rank = 1 
                rank_2 = 1             
                                               
                while len((s_1.cpu()[0])) == len((s_0.cpu()[0])) and done == False:
                    if len(probs_target.argsort().cpu().tolist()) - rank > 0:
                    
                        action_target = probs_target.argsort().cpu().tolist()[-rank]
                        indices = [i for i, x in enumerate(s_0_s[0]) if x == action]
                        filter_source = list(pd.Series(s_0_s.cpu()[1])[indices])
                        
                        values = range(0, len(filter_source))
                        map_ind = dict(zip(values, filter_source))
        
                        node_id_target = map_ind[action_target]
                      
                        for t in range(len(s_0_s[0])):
                          
                           if s_0_s[0][t]== node_id and s_0_s[1][t]== node_id_target:
                              edge_id = t
                        
                        s_1, done = env.step(s_0, edge_id, node_idx, stop_action_sampled)
                        
                        rank = rank + 1
               

                    else:
                        action = probs.argsort().cpu().tolist()[-rank_2]
                        stop_action_sampled = action == features_s.shape[0] 
                    
                        
                        node_id = action
                        probs_target = policy_trained(features_s, s_0_s, source=action)

                        if not stop_action_sampled:
                            action_target = probs_target.argmax().item()
                                           
                            indices = [i for i, x in enumerate(s_0_s[0]) if x == action]
                            filter_source = list(pd.Series(s_0_s.cpu()[1])[indices])
                            
                            values = range(0, len(filter_source))
                            map_ind = dict(zip(values, filter_source))
            
                            node_id_target = map_ind[action_target]
                            
                            for t in range(len(s_0_s[0])):
                              
                               if s_0_s[0][t]== node_id and s_0_s[1][t]== node_id_target:
                                  edge_id = t
                        
     
                        s_1, done = env.step(s_0, edge_id, node_idx, stop_action_sampled)
                        rank_2 = rank_2 + 1

                s_0 = s_1

            if done:

                final_edge_index = s_1
                selected_nodes = [dictr[x] for x in np.unique(s_1.cpu()[0])]
                if graph_mode:
                   node_idx_new = node_idx
                else:
                   node_idx_new = dictr[node_idx]
                np.unique(s_1.cpu()[0])
                times = times + 1
                edge_masks = final_edge_index
                
                adj_start = to_dense_adj(s_0_start)
                logits, adj_att = self.model(features_start, adj_start)
               
                probs_logits = F.softmax(logits, dim=-1)
                pred_labels = probs_logits.argmax(dim=-1)
                
                if graph_mode:                   
                    label = pred_labels.squeeze()
                    probs_logits = probs_logits.squeeze()
                
                else:
                    label = pred_labels.squeeze()[node_idx_new]
                    probs_logits = probs_logits.squeeze()[node_idx_new]
              
                data = Data(x=features_start, edge_index=s_0_start)
               
                masked_nodes_list = [node for node in range(data.x.shape[0]) if node in selected_nodes]
                maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

                
                if graph_mode:
                   value_func = GnnNetsNC2valueFuncGraph(self.model,
                                                    node_idx_new,
                                                    target_class=label)               
                
                else:
                   value_func = GnnNetsNC2valueFunc(self.model,
                                                    node_idx_new,
                                                    target_class=label)

                adj = to_dense_adj(s_0_start)
                log = self.model(features_start, adj)
                
                masked_pred = gnn_score(masked_nodes_list, features_start, s_0_start,
                                        value_func=value_func,
                                        subgraph_building_method='zero_filling')
                maskout_pred = gnn_score(maskout_nodes_list, features_start, s_0_start,
                                         value_func=value_func,
                                         subgraph_building_method='zero_filling')
                sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
                
                related_preds = [{'masked': masked_pred,
                                      'maskout': maskout_pred,
                                      'origin': probs_logits[label],
                                      'sparsity': sparsity_score,
                                      'final_edge_index':s_1,
                                      'node_idx': node_idx,
                                      'node_idx_new':node_idx,
                                      'syn': self.dataset}]
                  
                edge_masks = final_edge_index
                edges = pd.DataFrame(    {
                        "source": final_edge_index.cpu().detach().numpy()[0],
                        "target": final_edge_index.cpu().detach().numpy()[1],
                        
                    }
                )
                
            return None, edge_masks, related_preds

    def train_policy(self, node_indices, args):
        """
        Train explainer policy in args for node_indices
        :param node_indices:
        :param args:
        :return:
        """
        number_episodes = args.number_episodes
        hops_policy = args.hops_policy
        num_hops = args.num_hops
        policy_model = args.policy_model
        batch_size = args.batch_size
        gamma = args.gamma
        lr = args.lr
        lr_value = args.lr_value
        dataset = args.dataset
        
        adj = self.adj
        net = self.model
        feat = self.feat
        labels = self.label.tolist()
        pred = self.pred
        graph_mode = self.graph_mode
        
        node_idx_list = []
        size_list = []
        edge_index_list = []
        features_list = []

        if graph_mode:
            for graph_idx in node_indices:
                  node_idx_list.append(graph_idx)
                  adj_new = adj[graph_idx].cpu().detach().numpy()

                  feat_new = feat[graph_idx, :].numpy()
                  G = nx.from_numpy_matrix(adj_new)
                  feat_dict = {i: {'feat': np.array(feat_new, dtype=np.float32)} for i in G.nodes()}
                  nx.set_node_attributes(G, feat_dict)
          
                  data = from_networkx(G)
                  edge_index = data.edge_index.long().to(self.device)
                  #data.num_classes = max(max(sublist) for sublist in labels) + 1
    
                  features = torch.from_numpy(feat_new.reshape(-1, feat_new.shape[-1])).float().to(self.device)
                  data.input_dim = features.shape[1]
                  edge_index_list.append(edge_index)
                  features_list.append(features)
                  size_list.append(len(edge_index[0]))
            
                        
        else:
        
            adj_new = adj[0]  # 1 x n x n -> n x n
            G = nx.from_numpy_matrix(adj_new)
            feat_dict = {i: {'feat': np.array(feat, dtype=np.float32)} for i in G.nodes()}
            nx.set_node_attributes(G, feat_dict)
    
            data = from_networkx(G)
            edge_index = data.edge_index.long().to(self.device)
            data.num_classes = max(max(sublist) for sublist in labels) + 1
    
            features = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).float().to(self.device)
            data.input_dim = features.shape[1]
    
            # for which nodes to train policy
            for t in node_indices:
                node_idx_list.append(t)
                size_list.append(len(subgraph(t, features, edge_index, hops_policy)[1][1]))
            edge_index_list.append(edge_index)
            features_list.append(features)

        index_max = np.argmax(size_list)

        entropy_class = Entropy(net, graph_mode)
        env = PolicyExplainerEnv(node_idx_list, edge_index_list, features_list, entropy_class, num_hops, hops_policy, self.args)
        policy = eval(policy_model)(args.input_dim, args.policy_hidden_dim, args.num_heads)


        rewards, policy_t, loss = reinforce(env,
                                                     policy,
                                                     number_episodes,
                                                     edge_index_list,
                                                     features_list,
                                                     batch_size,
                                                     gamma,
                                                     lr,
                                                     lr_value,
                                                     hops_policy,
                                                     graph_mode)
       
        reward = "_782"
        version = str(dataset)+ "_" + str(number_episodes) + "_" + reward
        torch.save(policy_t.state_dict(), "log/models/policy_exp_" + version + ".pth")

        window = 10
        smoothed_rewards = [np.mean(rewards[i - window:i + 1]) if i > window
                            else np.mean(rewards[:i + 1]) for i in range(len(rewards))]

        plt.switch_backend("agg")
        plt.figure(figsize=(12, 8))
        plt.plot(rewards)
        plt.plot(smoothed_rewards)
        plt.ylabel('Mean Rewards')
        plt.xlabel('Episodes')
        plt.savefig("log/policy/rewards_exp_" + version + ".png")
        plt.close()

        plt.switch_backend("agg")
        plt.figure(figsize=(12, 8))
        plt.plot(loss, label = "Loss")
        plt.ylabel('Loss and Entropy')
        plt.xlabel('Episodes')
        plt.legend(loc='best')
        plt.savefig("log/policy/lossentropy" + version + ".png")
        plt.close()

        return policy_t
