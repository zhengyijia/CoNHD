import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import os
import sys
from tqdm import tqdm
import time
import dgl
from scipy.sparse import vstack as s_vstack
from sklearn.preprocessing import StandardScaler

from utils import args_parser
from utils import metric
import utils.data_load as dl

from model.layer import FC, Wrap_Embedding

from utils.sampler import HypergraphNeighborSampler
from model.CoNHD_GD import CoNHD_GD, CoNHD_GD_Layer, CoNHDScorer
from model.CoNHD_ADMM import CoNHD_ADMM, CoNHD_ADMM_Layer

# Make Output Directory --------------------------------------------------------------------------------------------------------------
args = args_parser.parse_args()
args.task_type = 'edge_dependent_node_classification'
if args.evaltype == "test":
    outputdir = "results_test/" + args.task_type + "/" + args.dataset_name + "/" + args.init_feat_type + "/"
    outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
    if args.recalculate is False and os.path.isfile(outputdir + "test_prediction.txt"):
        sys.exit("Already Run")
else:
    outputdir = "results/" + args.task_type + "/" + args.dataset_name + "/" + args.init_feat_type + "/"
    outputdir += args.model_name + "/" + args.param_name +"/"
    if args.recalculate is False and os.path.isfile(outputdir + "test_prediction.txt"):
        sys.exit("Already Run")
if os.path.isdir(outputdir) is False:
    sys.exit(f"Model folder not found: {outputdir}")
print("OutputDir = " + outputdir)

if not os.path.isfile(outputdir + "initembedder.pt"): 
    sys.exit(f"initembedder.pt not found! ")
if not os.path.isfile(outputdir + "embedder.pt"):
    sys.exit(f"embedder.pt not found! ")
if not os.path.isfile(outputdir + "scorer.pt"):
    sys.exit(f"scorer.pt not found! ")

if args.recalculate and os.path.isfile(outputdir + "test_prediction.txt"):
    os.remove(outputdir + "test_prediction.txt") 

# Initialization --------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset_name = args.dataset_name #'citeseer' 'cora'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
dgl.seed(args.seed)

# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args)
test_data = data.get_data(2)
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = dl.gen_weighted_DGLGraph(data.hedge2node, data.hedge2nodePE, data.hedge2nodepos)
else:
    g = dl.gen_DGLGraph(data.hedge2node, data.hedge2nodepos)
fullsampler = HypergraphNeighborSampler(full_ls)

if args.use_gpu:
    g = g.to(device)
    test_data = torch.tensor(test_data, dtype=torch.long).to(device)
    data.e_feat = data.e_feat.to(device)

# ensure corresponding edges in 'in' and 'con' have same ids
assert (g.edge_ids(*g.edges(etype='in'), etype='in') 
        == g.edge_ids(*reversed(g.edges(etype='in')), etype='con')).all(), \
        "corresponding \'in\' and \'con\' edges should have the same ids! "

testdataloader = dgl.dataloading.NodeDataLoader(g, {"edge": test_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False)

args.input_edim = data.e_feat.size(1)
args.order_dim = data.order_dim

# init embedder
args.input_vdim = args.init_feat_dim
init_feat_fname = f'{args.feature_inputdir}{args.task_type}/{args.init_feat_type}/{args.dataset_name}_dim_{args.init_feat_dim}.npy'
feat_origin_index_fname = f'{args.feature_inputdir}{args.task_type}/{args.init_feat_type}/{args.dataset_name}_dim_{args.init_feat_dim}_origin_index.txt'
node_list = np.arange(data.numnodes).astype('int')
print("load exist init features: ")
print(init_feat_fname, flush=True)
A = np.load(init_feat_fname)
# Sort feature matrix according to node indexes
with open(feat_origin_index_fname, 'r') as f: 
    feat_origin_indexes = f.readlines()
    feat_origin_indexes = dict(zip([index.strip() for index in feat_origin_indexes], list(np.arange(len(feat_origin_indexes)))))
sort_indexes = np.array([feat_origin_indexes[data.node_orgindex[i]] for i in range(data.numnodes)])
A = A[sort_indexes, :]
# feature transform
A = StandardScaler().fit_transform(A)  ##
A = A.astype('float32')
A = torch.tensor(A).to(device)
initembedder = Wrap_Embedding(data.numnodes, args.input_vdim, scale_grad_by_freq=False, padding_idx=0, sparse=False)
initembedder.weight = nn.Parameter(A)

print("Model:", args.embedder)
# model init
if args.embedder == "CoNHD_GD": 
    embedder = CoNHD_GD(CoNHD_GD_Layer, args.PE_Block, args.input_vdim, args.co_rep_dim, 
                        weight_dim=args.order_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                        num_inds=args.num_inds, att_type_v=args.att_type_v, att_type_e=args.att_type_e, 
                        num_att_layer=args.num_att_layer, dropout=args.dropout, input_dropout=args.input_dropout, 
                        weight_flag=data.weight_flag, layernorm=args.layernorm, 
                        node_agg=args.node_agg, hedge_agg=args.hedge_agg).to(device)
elif args.embedder == "CoNHD_ADMM": 
    embedder = CoNHD_ADMM(CoNHD_ADMM_Layer, args.PE_Block, args.input_vdim, args.co_rep_dim, 
                          weight_dim=args.order_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                          num_inds=args.num_inds, att_type_v=args.att_type_v, att_type_e=args.att_type_e, 
                          num_att_layer=args.num_att_layer, dropout=args.dropout, input_dropout=args.input_dropout, 
                          weight_flag=data.weight_flag, layernorm=args.layernorm).to(device)

print("Embedder to Device")
print("Scorer = ", args.scorer)
# pick scorer
if args.scorer == "sm":
    scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers).to(device)
elif args.scorer == "im": #whatsnet
    if args.embedder == "CoNHD_GD" or args.embedder == "CoNHD_ADMM": 
        scorer = CoNHDScorer(args.output_dim, args.co_rep_dim, dim_hidden=args.dim_hidden, num_layers=args.scorer_num_layers).to(device)

initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device

initembedder.eval()
embedder.eval()
scorer.eval()

with torch.no_grad():
    total_pred = []
    total_label = []
    total_node_indexes = []
    total_hedge_indexes = []
    num_data = 0
    total_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    for input_nodes, output_nodes, blocks in testdataloader:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices = srcs.to(device)
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)

        node_indexes = input_nodes['node'][nodeindices]
        hedge_indexes = input_nodes['edge'][hedgeindices]

        batchcount += 1
        # Get Embedding
        if args.embedder == "CoNHD_GD" or args.embedder == "CoNHD_ADMM": 
            v_feat = initembedder(input_nodes['node'].to(device))
            co_feat, co_eid = embedder(blocks[(len(blocks)//2):], v_feat)
        else:
            v_feat = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat)
                
        # Predict Class
        if args.scorer == "sm":
            hembedding = e[hedgeindices]
            vembedding = v[nodeindices]
            input_embeddings = torch.cat([hembedding,vembedding], dim=1)
            predictions = scorer(input_embeddings)
        elif args.scorer == "im":
            if args.embedder == "CoNHD_GD" or args.embedder == "CoNHD_ADMM": 
                predictions, nodelabels, _, _ = scorer(blocks[-1], co_feat, co_eid)
            else: 
                predictions, nodelabels, node_indexes, hedge_indexes = scorer(blocks[-1], v, e)
        total_pred.append(predictions.detach().cpu())
        total_label.append(nodelabels.detach().cpu())
        total_node_indexes.append(node_indexes.detach().cpu())
        total_hedge_indexes.append(hedge_indexes.detach().cpu())
        
        num_data += predictions.shape[0]

    total_pred = torch.cat(total_pred, dim=0).numpy()
    total_label = torch.cat(total_label, dim=0).numpy()
    total_pred_cls = np.argmax(total_pred, axis=1)
    total_node_indexes = torch.cat(total_node_indexes, dim=0).numpy()
    total_hedge_indexes = torch.cat(total_hedge_indexes, dim=0).numpy()
    total_node_names = [data.node_orgindex[idx] for idx in total_node_indexes]
    total_hedge_names = [data.hedgename[idx] for idx in total_hedge_indexes]

    pred_results = list(zip(total_hedge_names, total_node_names, 
                            total_label, total_pred_cls, *(zip(*(total_pred.tolist())))))
    pred_results = ['\t'.join([str(item) for item in instance])+'\n' for instance in pred_results]
    with open(outputdir + "test_prediction.txt", "w") as f: 
        f.write(f'hedge\tnode\tlabel\tpred\t' + 
                '\t'.join([f'pred_prob_{i}' for i in range(total_pred.shape[1])]) + '\n')
        f.writelines(pred_results)

print('Write predict results success. ', flush=True)

