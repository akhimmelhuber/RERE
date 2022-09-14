""" 
     Main user interface for the explainer module RERE.


"""



import os
from tensorboardX import SummaryWriter
import shutil
import scipy.stats
from explainer.experiment import *
from explainer import models
from explainer import reinforce
import warnings
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import argparse
import networkx as nx
from torch_geometric.utils import from_networkx, sort_edge_index
from explainer.utils import k_hop_subgraph, discount_rewards
from openpyxl import load_workbook
from utils import graph_utils
from metrics import XCollector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden_dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--input_dim", dest="input_dim", type=int, help="Input dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain_node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
 

    parser.add_argument(
        "--number_episodes",
        dest="number_episodes",
        help="train policy for number of episodes"
    )

    parser.add_argument(
        "--num_hops",
        dest="num_hops",
        help="number of hops around center node"
    )

    parser.add_argument(
        "--hops_policy",
        dest="hops_policy",
        help="number of hops for the policy network"
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    parser.add_argument(
        "--policy_model",
        dest="policy_model",
        help="model class of the policy network"
    )

    parser.add_argument(
        "--policy_hidden_dim",
        dest="policy_hidden_dim",
        help="model class of the policy network"
    )

    parser.add_argument(
         "--constraint", dest="constraint", type=int, help="Size constraint(min)"
     )
    parser.add_argument(
         "--plot_graphs", dest="constraint", type=int, help="Plot Sequential Graphs"
     )


    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        opt="adam",  
        opt_scheduler="none",
        cuda="0",
        clip=2.0,
        hidden_dim=20,
        num_epochs=100,
        output_dim=20,
        num_gc_layers=3,
        name_suffix="",
        explainer_suffix="",
        dropout=0.0,
        method="base",
        align_steps=1000,
        # policy arguments
        dataset="syn1",
        number_episodes=41000,
        policy_model="PolicyNetworkRescal",
        gamma=0.99,
        batch_size=5,
        num_hops=3,
        hops_policy=3,
        lr=1e-4,
        lr_value=1e-4,
        input_dim=10,
        policy_hidden_dim=10,
        num_heads=2,
        explain_node=1,    
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
        constraint = 1,
        plot_graphs = False,
    )


    return parser.parse_args()


def main():
    # Load a configuration
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]    # get computation graph
    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    # build model
    print("Method: ", prog_args.method)
    if graph_mode: 
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    else:
        # Explain Node prediction
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
    adj=cg_dict["adj"]
    feat=cg_dict["feat"]
    label=cg_dict["label"]
    pred=cg_dict["pred"]
    train_idx=cg_dict["train_idx"]
    neighborhoods = None if graph_mode else graph_utils.neighborhoods(adj=adj,n_hops=3,use_cuda=False)
      
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


    # Create explainer
    
    experiment = Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        device=device,
        writer=writer,
        print_training=True,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
    )

    if prog_args.explain_node is None:

       if graph_mode:
            if prog_args.graph_idx == 3:
                # Train Policy for Graph Classification GCN
                experiment.train_policy([10], prog_args)
           
       else:       
                # Train Policy for Node Classification GCN
                masked_adj = experiment.train_policy([312], prog_args)
                
    
    else:
                        
       if prog_args.dataset == "syn1":
          ckpt_path = "ckpt/syn1_base_h20_o20.pth.tar"
          policy = "log/models/policy_exp_syn1_30000__diff_469.pth"
          node_indices = range(300, 320, 1)
          # node_indices = [322]

       elif prog_args.dataset == "syn4":
          ckpt_path = "ckpt/syn4_base_h20_o20.pth.tar"

          policy = "log/models/policy_exp_syn4_30000__diff_607.pth"
          node_indices = range(511, 872, 1) 
          node_indices=[850]

       elif prog_args.dataset == "syn5":
          ckpt_path = "ckpt/syn5_base_h20_o20.pth.tar"
         
          policy = "log/models/policy_exp_syn5_30000__diff_530.pth"
          node_indices = range(511, 1231, 1)
          node_indices = [550]
          
       elif prog_args.dataset == "Mutagenicity":
          ckpt_path = "ckpt/Mutagenicity_base_h20_o20.pth.tar"
          #NonMutag
          #policy = "log/models/policy_exp_Mutagenicity_150000__diff_nonmutag44.pth"
          node_indices = [1,2]          
          #Mutag
          policy = "log/models/policy_exp_Mutagenicity_150000__diff_mutag30.pth"
       
       elif prog_args.dataset == "REDDIT-BINARY":
              node_indices = range(0,400,1)
              policy ="log/models/policy_exp_REDDIT-BINARY_9000__77.pth" 
           
       x_collector = XCollector()
       index = -1
       
       for j, node_idx in enumerate(node_indices):
           index += 1
           graph_idx = node_idx
          
           if graph_mode:
               
               adj_new = adj[graph_idx].cpu().detach().numpy()
       
               feat_new = feat[graph_idx, :].numpy()
               G = nx.from_numpy_matrix(adj_new)
               feat_dict = {i: {'feat': np.array(feat_new, dtype=np.float32)} for i in G.nodes()}
               nx.set_node_attributes(G, feat_dict)
       
               data = from_networkx(G)

               edge_index = data.edge_index.long().to(device)
               x = torch.from_numpy(feat_new.reshape(-1, feat_new.shape[-1])).float().to(device)
               data.input_dim = x.shape[1]
               
       
           else:
              
               node_idx_new, sub_adj, sub_feat, sub_label, neighbors = extract_neighborhood(
                   node_idx, graph_idx
               )
               sub_label = np.expand_dims(sub_label, axis=0)
               adj_new = adj.reshape(-1, adj.shape[-1])
               
               G = nx.from_numpy_matrix(adj_new)
               feat_dict = {i: {'feat': np.array(feat, dtype=np.float32)} for i in G.nodes()}
               nx.set_node_attributes(G, feat_dict)
               
               data = from_networkx(G)
               edge_index = data.edge_index.long().to(device)
               
               x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).float().to(device)
             
           with torch.no_grad():
                          walks, masks, related_preds = experiment.explain(policy, graph_mode, x, edge_index, prog_args.constraint, prog_args.plot_graphs,  node_idx=node_idx)
   
           masks = [mask.detach() for mask in masks]
          
           x_collector.collect_data(masks, related_preds)
       
       
       
           
       print(f'Fidelity: {x_collector.fidelity:.4f}'
             f'Fidelity_inv: {x_collector.fidelity_inv:.4f}'
             f'Sparsity: {x_collector.sparsity:.4f}')
             
       if graph_mode ==False:
            print(f'F1_Score: {x_collector.f1_score_ba:.4f}')
     
     
      
if __name__ == "__main__":
    main()




