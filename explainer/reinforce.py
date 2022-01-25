import numpy as np
import torch
from explainer.utils import k_hop_subgraph, discount_rewards, gnnx_result
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch.nn as nn
import warnings
import pandas as pd

EPS = 1e-15

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
torch.set_printoptions(threshold=10000)


def reinforce(env, policy_estimator, num_episodes, edge_index_list, features_list, batch_size, gamma, lr, lr_value, hops_policy, graph_mode):
    """
    REINFORCE policy gradient algorithm
    """
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_features = []
    batch_actions_target = []
    batch_probs = []
    batch_done = []
    batch_done_counter = []
    batch_probs_target =[]
    batch_actions_dict = []
    batch_states = []
    batch_features = []
    batch_reward_filter = []
    batch_advantage = []
    batch_advantage_target = []
    
    exclude_node_list = []    
    batch_nodeidx =[]
    batch_counter = 1 
    ep = 0
    
    loss_vec = []
    avg_entropy = []
    
    index = []
    target_counter = 0
    done_counter = 0
    

    optimizer = torch.optim.Adam(policy_estimator.parameters(), lr=lr)
    optimizer_value = torch.optim.Adam(policy_estimator.parameters(), lr=lr_value)
    optimizer.zero_grad()
    policy_estimator.train()

    while ep < num_episodes:
        print("Episode", ep)
        s_0, node_idx = env.reset(edge_index_list)
        features = features_list[0]
        action_space = np.unique(s_0.cpu()).tolist()
        states = []
        feature = []
        policy_estimator_results = []
        policy_estimator_results_target = []
        node_idx_list = []
        rewards = []
        actions = []
        actions_target = []
        advantage = []
        advantage_target = []
        reward_filter = []
        done_list = []
        actions_dict = []
        done = False
        exclude_node = []
        entropy_vec = []

        if ep > 0:
           if steps >= 500:
              done_counter = len(batch_rewards)
              print("steps", steps)
        
        #print("done counter", done_counter)
        max_steps = 500
        steps = 0

        # episode loops
        while done == False and steps < max_steps:
      
            if graph_mode:
              
                num_nodes = s_0.max().item() + 1
                subset = np.unique(s_0[0])

                row, col = s_0
                relabelled = row.new_full((num_nodes, ), -1)
                relabelled[subset] = torch.arange(len(subset), device=row.device)
                s_0_s = relabelled[s_0]
                features_s = features[subset]


            
            else: 
            
                subsetx, s_0, edge_maskx = k_hop_subgraph(node_idx, hops_policy, s_0, relabel_nodes=False, num_nodes=None,
                                                          flow='source_to_target')
                subset, s_0_s, edge_mask = k_hop_subgraph(node_idx, hops_policy, s_0, relabel_nodes=True, num_nodes=None,
                                                          flow='source_to_target')
                                                          
                        
                features_s = features[subset]                                  

            
            probs = policy_estimator(features_s, s_0_s, source=None)
            m = Categorical(probs)
            action_sample = m.sample()
            action = action_sample.item()
            stop_action_sampled = action == features_s.shape[0]
           
            
            vf_t = policy_estimator.value_function(features_s, s_0_s, source=None)
            advantage.append(vf_t)

            if not stop_action_sampled:
                node_id = action
                probs_target = policy_estimator(features_s, s_0_s, source=action)
                m_target = Categorical(probs_target)
                action_sample_target = m_target.sample()
                action_target = action_sample_target.item()
                

                indices = [i for i, x in enumerate(s_0_s[0]) if x == action]
                filter_source = list(pd.Series(s_0_s[1])[indices])
                
                values = range(0, len(filter_source))
                map_ind = dict(zip(values, filter_source))

                node_id_target = map_ind[action_target]
                

                for x in range(len(s_0_s[0])):
                  
                   if s_0_s[0][x]== node_id and s_0_s[1][x]== node_id_target:
                      edge_id = x

                policy_estimator_results_target.append(probs_target[action_target])

                vf_t_target = policy_estimator.value_function(features_s, s_0_s, source=action)
                advantage_target.append(vf_t_target)

            else:
               
                edge_id = None
                # if stop action was not the immediate first action
                reward_filter.append(done_counter)

            s_1, done = env.step(s_0, edge_id, node_idx, stop_action_sampled)
            steps += 1
            target_counter += 1
        
            if done:
                 r = env.reward_final(s_1, s_0, node_idx)
                 
            else:
                 r = env.reward(s_1, s_0, node_idx)

            policy_estimator_results.append(probs[action])
            exclude_node_list.append(exclude_node[:])
            if not stop_action_sampled:
                exclude_node.append(action)
                
            done_list.append(done)
            states.append(s_0)
            feature.append(features_s)
            rewards.append(r)
            
            done_counter += 1

            if not stop_action_sampled:
                actions.append(action)
                actions_target.append(node_id_target)
            else:
                actions.append(-1)
            actions_dict.append(action)
            node_idx_list.append(node_idx)

            s_0 = s_1

            # If done, batch data
            if done:
                batch_probs.extend(policy_estimator_results)
                batch_probs_target.extend(policy_estimator_results_target)
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_reward_filter.extend(reward_filter)

                batch_states.extend(states)
                batch_features.extend(feature)
                batch_actions.extend(actions)
                batch_actions_target.extend(actions_target)
                batch_done.extend(done_list)
                batch_nodeidx.extend(node_idx_list)
                batch_advantage.extend(advantage)
                batch_advantage_target.extend(advantage_target)

                batch_actions_dict.extend(actions_dict)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                exclude_node = []

                if batch_counter == batch_size:
                    state_tensor = batch_states
                    feature_tensor = batch_features
                    reward_tensor = torch.FloatTensor(batch_rewards)                  
                    action_tensor = torch.LongTensor(batch_actions)
                    advantage_tensor = torch.cat(batch_advantage)
                    # if first action is always stop action there are no target actions
                    if len(batch_advantage_target) > 0:
                        advantage_target_tensor = torch.cat(batch_advantage_target)
                    else:
                        advantage_target_tensor = torch.tensor([])

                    action_tensor_target = torch.LongTensor(batch_actions_target)
                    
                    nodeidx_tensor = batch_nodeidx

                    probs = torch.stack(batch_probs)
                    logprob = torch.log(probs+EPS).cpu()  
                    
                    with torch.no_grad():
                              advantage = reward_tensor - advantage_tensor  
        

                    selected_logprobs = advantage.detach() * -logprob
                    loss = selected_logprobs.mean()
                    
 
                    if len(batch_probs_target) > 0:
                      probs_target = torch.stack(batch_probs_target)
                      logprob_target = torch.log(probs_target+EPS).cpu()  
                      reward_tens = torch.tensor([v for i,v in enumerate(reward_tensor.cpu()) if i not in batch_reward_filter])

                      with torch.no_grad():
                              advantage_t = reward_tens - advantage_target_tensor  

                      selected_logprobs_target = advantage_t.detach() * -logprob_target
                      loss_target = selected_logprobs_target.mean()
                    else:
                       loss_target = 0

                    loss = (loss + loss_target)
                    loss_vec.append(loss.item())

                    # Calculate gradients
                    loss.backward()
                    
                    # calculate vf loss
                    loss_fn = nn.MSELoss()
                    vf_loss_source = loss_fn(advantage_tensor, reward_tensor)
                    if len(batch_probs_target) > 0:
                        vf_loss_target = loss_fn(advantage_target_tensor, reward_tens)
                    else:
                        vf_loss_target = 0
                    vf_loss = vf_loss_source + vf_loss_target
                    vf_loss.backward()

                    # Apply gradients
                    optimizer.step()
                    optimizer.zero_grad()

                    optimizer_value.step() 
                    optimizer_value.zero_grad()

                    batch_rewards = []
                    batch_advantage = []
                    batch_advantage_target = []
                    batch_reward_filter = []
                    batch_actions = []
                    batch_actions_target = []
                    batch_states = []
                    exclude_node_list = []
                    batch_nodeidx = []
                    batch_probs = []
                    batch_probs_target = []
                    reward_filter = []
                    target_counter = 0
                    done_counter = 0
                    
                    batch_counter = 1
                   
                if ep >= 100:
                    
                    print("Rolling mean reward:", np.mean(total_rewards[-100:]))
                    print("Rolling loss:", np.mean(loss_vec[-100:]))
                    print("Avg. episode length: ", len(batch_rewards) / batch_counter)

                ep += 1

    return total_rewards, policy_estimator, loss_vec
