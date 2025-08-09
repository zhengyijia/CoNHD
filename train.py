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

def run_epoch(args, data, dataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train"):
    total_pred = []
    total_label = []
    num_data = 0
    total_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    for input_nodes, output_nodes, blocks in dataloader:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices = srcs.to(device)
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
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
                predictions, nodelabels, _, _ = scorer(blocks[-1], v, e)
        total_pred.append(predictions.detach())
        total_label.append(nodelabels.detach())
        
        # Back Propagation
        num_data += predictions.shape[0]
        loss = loss_fn(predictions, nodelabels)
        if opt == "train":
            optim.zero_grad()
            loss.backward() 
            # TODO: clip gradient
            # grad_norm = torch.nn.utils.clip_grad_norm_(initembedder.parameters(), max_norm=1)
            # print("initembedder, 95th percentile of gradient norm:", torch.quantile(grad_norm, 0.95), flush=True)
            # grad_norm = torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1)
            # print("embedder, 95th percentile of gradient norm:", torch.quantile(grad_norm, 0.95), flush=True)
            # grad_norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1)
            # print("scorer, 95th percentile of gradient norm:", torch.quantile(grad_norm, 0.95), flush=True)
            torch.nn.utils.clip_grad_norm_(initembedder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1)
            optim.step()
        total_loss += (loss.item() * predictions.shape[0])
        if opt == "train":
            torch.cuda.empty_cache()
    
    print("Time : ", time.time() - ts)
    
    return total_pred, total_label, total_loss / num_data, initembedder, embedder, scorer, optim, scheduler

def run_test_epoch(args, data, testdataloader, initembedder, embedder, scorer, loss_fn):
    total_pred = []
    total_label = []
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
                predictions, nodelabels, _, _ = scorer(blocks[-1], v, e)
        total_pred.append(predictions.detach())
        pred_cls = torch.argmax(predictions, dim=1)
        total_label.append(nodelabels.detach())
        
        num_data += predictions.shape[0]
        loss = loss_fn(predictions, nodelabels)

        total_loss += (loss.item() * predictions.shape[0])
        
    return total_pred, total_label, total_loss / num_data, initembedder, embedder, scorer

# Make Output Directory --------------------------------------------------------------------------------------------------------------
args = args_parser.parse_args()
if args.task_type is None: 
    args.task_type = 'edge_dependent_node_classification'
if args.evaltype == "test":
    outputdir = "results_test/" + args.task_type + "/" + args.dataset_name + "/" + args.init_feat_type + "/"
    outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
    if args.recalculate is False and os.path.isfile(outputdir + "log_test_confusion.txt"):
        sys.exit("Already Run")
else:
    outputdir = "results/" + args.task_type + "/" + args.dataset_name + "/" + args.init_feat_type + "/"
    outputdir += args.model_name + "/" + args.param_name +"/"
    if args.recalculate is False and os.path.isfile(outputdir + "log_test_confusion.txt"):
        sys.exit("Already Run")
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
print("OutputDir = " + outputdir)

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

valid_epoch = args.valid_epoch

if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    print("Start from checkpoint")
elif (args.recalculate is False and args.evaltype == "valid") and os.path.isfile(outputdir + "log_valid_micro.txt"):
    max_acc = 0
    cur_patience = 0
    epoch = 0
    with open(outputdir + "log_valid_micro.txt", "r") as f:
        for line in f.readlines():
            ep_str = line.rstrip().split(":")[0].split(" ")[0]
            acc_str = line.rstrip().split(":")[-1]
            epoch = int(ep_str)
            if max_acc < float(acc_str):
                cur_patience = 0
                max_acc = float(acc_str)
            else:
                cur_patience += 1
            if cur_patience > args.patience:
                break
    if cur_patience > args.patience or epoch == args.epochs:
        sys.exit("Already Run by log valid micro txt")
else:
    if os.path.isfile(outputdir + "log_train.txt"):
        os.remove(outputdir + "log_train.txt")
    if os.path.isfile(outputdir + "log_valid_micro.txt"):
        os.remove(outputdir + "log_valid_micro.txt")
    if os.path.isfile(outputdir + "log_valid_confusion.txt"):
        os.remove(outputdir + "log_valid_confusion.txt")
    if os.path.isfile(outputdir + "log_valid_macro.txt"):
        os.remove(outputdir + "log_valid_macro.txt")
    if os.path.isfile(outputdir + "log_test_micro.txt"):
        os.remove(outputdir + "log_test_micro.txt")
    if os.path.isfile(outputdir + "log_test_confusion.txt"):
        os.remove(outputdir + "log_test_confusion.txt")
    if os.path.isfile(outputdir + "log_test_macro.txt"):
        os.remove(outputdir + "log_test_macro.txt")
        
    if os.path.isfile(outputdir + "initembedder.pt"):
        os.remove(outputdir + "initembedder.pt")
    if os.path.isfile(outputdir + "embedder.pt"):
        os.remove(outputdir + "embedder.pt")
    if os.path.isfile(outputdir + "scorer.pt"):
        os.remove(outputdir + "scorer.pt")
    if os.path.isfile(outputdir + "evaluation.txt"):
        os.remove(outputdir + "evaluation.txt")
            
# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args)
train_data = data.get_data(0)
valid_data = data.get_data(1)
if args.evaltype == "test":
    test_data = data.get_data(2)
ls = [{('node', 'in', 'edge'): args.node_sampling, ('edge', 'con', 'node'): args.hedge_sampling}] * (args.num_layers * 2) + \
    [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}]  # do not sample the last layer (node-hedge pairs)
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = dl.gen_weighted_DGLGraph(data.hedge2node, data.hedge2nodePE, data.hedge2nodepos)
else:
    g = dl.gen_DGLGraph(data.hedge2node, data.hedge2nodepos)

fullsampler = HypergraphNeighborSampler(full_ls)
sampler = HypergraphNeighborSampler(ls)

if args.use_gpu:
    g = g.to(device)
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    if args.evaltype == "test":
        test_data = test_data.to(device)
    data.e_feat = data.e_feat.to(device)
    
# ensure corresponding edges in 'in' and 'con' have same ids
assert (g.edge_ids(*g.edges(etype='in'), etype='in') 
        == g.edge_ids(*reversed(g.edges(etype='in')), etype='con')).all(), \
        "corresponding \'in\' and \'con\' edges should have the same ids! "

dataloader = dgl.dataloading.NodeDataLoader( g, {"edge": train_data}, sampler, 
                                            batch_size=len(train_data) if args.bs==-1 else args.bs, 
                                            shuffle=True, drop_last=False) # , num_workers=4
validdataloader = dgl.dataloading.NodeDataLoader(g, {"edge": valid_data}, sampler, 
                                                 batch_size=len(valid_data) if args.bs==-1 else args.bs, 
                                                 shuffle=True, drop_last=False)
if args.evaltype == "test":
    testdataloader = dgl.dataloading.NodeDataLoader(g, {"edge": test_data}, fullsampler, 
                                                    batch_size=len(test_data) if args.bs==-1 else args.bs, 
                                                    shuffle=False, drop_last=False)

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
                        node_agg=args.node_agg, hedge_agg=args.hedge_agg, share_weights=args.share_weights).to(device)
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
    
params = []  
if args.embedder == "unigcnii":
    if not args.fix_init_embedder: 
        params += list(initembedder.parameters())
    params += list(scorer.parameters())
    optim = torch.optim.Adam([
            dict(params=embedder.reg_params, weight_decay=0.01),
            dict(params=embedder.non_reg_params, weight_decay=5e-4),
            dict(params=params, weight_decay=0.0)
        ], lr=0.01)
elif args.optimizer == "adam":
    if not args.fix_init_embedder: 
        params += list(initembedder.parameters())
    params += list(embedder.parameters())
    params += list(scorer.parameters())
    # TODO: eps
    # optim = torch.optim.Adam(params, lr=args.lr) #, weight_decay=args.weight_decay)
    optim = torch.optim.Adam(params, lr=args.lr, eps=1e-4, weight_decay=1e-4) #, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    if not args.fix_init_embedder: 
        params += list(initembedder.parameters())
    params += list(embedder.parameters())
    params += list(scorer.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)
elif args.optimizer == "rms": 
    if not args.fix_init_embedder: 
        params += list(initembedder.parameters())
    params += list(embedder.parameters())
    params += list(scorer.parameters())
    optim = torch.optim.RMSprop(params, lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
loss_fn = nn.CrossEntropyLoss()

# if args.embedder == "unigcnii":
#     optim = torch.optim.Adam([
#             dict(params=embedder.reg_params, weight_decay=0.01),
#             dict(params=embedder.non_reg_params, weight_decay=5e-4),
#             dict(params=list(initembedder.parameters()) + list(scorer.parameters()), weight_decay=0.0)
#         ], lr=0.01)
# elif args.optimizer == "adam":
#     optim = torch.optim.Adam(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr) #, weight_decay=args.weight_decay)
# elif args.optimizer == "adamw":
#     optim = torch.optim.AdamW(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
# elif args.optimizer == "rms":
#     optime = torch.optim.RMSprop(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
# loss_fn = nn.CrossEntropyLoss()

# Train =================================================================================================================================================================================
train_acc=0
patience = 0
best_eval_acc = 0
best_eval_epoch = 0
epoch_start = 1
if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    checkpoint = torch.load(outputdir + "checkpoint.pt") #, map_location=device)
    epoch_start = checkpoint['epoch'] + 1
    initembedder.load_state_dict(checkpoint['initembedder'])
    embedder.load_state_dict(checkpoint['embedder'])
    scorer.load_state_dict(checkpoint['scorer'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_eval_acc = checkpoint['best_eval_acc']
    best_eval_epoch = checkpoint['best_eval_epoch']
    patience = checkpoint['patience']    
    
    print("Load {} epoch trainer".format(epoch_start))
    print("best_eval_acc = {}\tpatience = {}".format(best_eval_acc, patience))
    
print("# Embedder Params:", sum(p.numel() for p in embedder.parameters() if p.requires_grad), flush=True)
with open(outputdir + "log_train.txt", "+a") as f:
    f.write(f'# Embedder Params: {sum(p.numel() for p in embedder.parameters() if p.requires_grad)}')
    f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
    f.write("Training start, epoch: %d\n" % (epoch_start))
epoch = epoch_start
for epoch in tqdm(range(epoch_start, args.epochs + 1), desc='Epoch'): # tqdm
    
    # break if patience from checkpoint is larger than the threshold
    if patience > args.patience:
        break
    
    print("Training")
    
    # Training stage
    initembedder.train()
    embedder.train()
    scorer.train()
    
    # Calculate Accuracy & Epoch Loss
    print("Start training epoch: %d" % (epoch))
    with open(outputdir + "log_train.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
        f.write("Start training epoch: %d\n" % (epoch))
    total_pred, total_label, train_loss, initembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, dataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train")
    with open(outputdir + "log_train.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
        f.write("End training epoch: %d\n" % (epoch))
    
    total_pred = torch.cat(total_pred)
    total_label = torch.cat(total_label, dim=0)
    pred_cls = torch.argmax(total_pred, dim=1)
    train_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    scheduler.step()
    print("%d epoch: Training loss : %.4f / Training acc : %.4f\n" % (epoch, train_loss, train_acc), flush=True)
    with open(outputdir + "log_train.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
        f.write("%d epoch: Training loss : %.4f / Training acc : %.4f\n" % (epoch, train_loss, train_acc))
        
    # Test ===========================================================================================================================================================================
    if epoch % valid_epoch == 0:
        initembedder.eval()
        embedder.eval()
        scorer.eval()
        
        with torch.no_grad():
            total_pred, total_label, eval_loss, initembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, validdataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="valid")
        # Calculate Accuracy & Epoch Loss
        total_label = torch.cat(total_label, dim=0)
        total_pred = torch.cat(total_pred)
        pred_cls = torch.argmax(total_pred, dim=1)
        eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
        y_test = total_label.cpu().numpy()
        pred = pred_cls.cpu().numpy()
        confusion, accuracy, precision, recall, f1 = metric.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_micro.txt", "+a") as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
            f.write("{} epoch:Test Loss:{} /Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, accuracy,precision,recall,f1))
        confusion, accuracy, precision, recall, f1 = metric.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_confusion.txt", "+a") as f:
            for r in range(args.output_dim):
                for c in range(args.output_dim):
                    f.write(str(confusion[r][c]))
                    if c == args.output_dim -1 :
                        f.write("\n")
                    else:
                        f.write("\t")
        with open(outputdir + "log_valid_macro.txt", "+a") as f:    
            f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))           
            f.write("{} epoch:Test Loss:{} /Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, accuracy,precision,recall,f1))

        if best_eval_acc < eval_acc:
            best_eval_acc = eval_acc
            best_eval_epoch = epoch
            print(best_eval_acc)
            patience = 0
            if args.evaltype == "test" or args.save_best_epoch:
                print("Model Save")
                modelsavename = outputdir + "embedder.pt"
                torch.save(embedder.state_dict(), modelsavename)
                scorersavename = outputdir + "scorer.pt"
                torch.save(scorer.state_dict(), scorersavename)
                initembeddersavename = outputdir + "initembedder.pt"
                torch.save(initembedder.state_dict(),initembeddersavename)
        else:
            patience += 1

        if patience > args.patience:
            break
        
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'scorer' : scorer.state_dict(),
            'initembedder' : initembedder.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'optimizer': optim.state_dict(),
            'best_eval_acc' : best_eval_acc,
            'best_eval_epoch': best_eval_epoch, 
            'patience' : patience
            }, outputdir + "checkpoint.pt")

if args.evaltype == "test":
    print("Test")
    print(f"best eval epoch: {best_eval_epoch}")
    
    initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
    embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
    scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device
    
    initembedder.eval()
    embedder.eval()
    scorer.eval()

    with open(outputdir + "log_test_micro.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, start testing\n',time.localtime(time.time())))
    with torch.no_grad():
        total_pred, total_label, test_loss, initembedder, embedder, scorer = run_test_epoch(args, data, testdataloader, initembedder, embedder, scorer, loss_fn)
    with open(outputdir + "log_test_micro.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, end testing\n',time.localtime(time.time())))
    # Calculate Accuracy & Epoch Loss
    total_label = torch.cat(total_label, dim=0)
    total_pred = torch.cat(total_pred)
    pred_cls = torch.argmax(total_pred, dim=1)
    eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    y_test = total_label.cpu().numpy()
    pred = pred_cls.cpu().numpy()
    confusion, accuracy, precision, recall, f1 = metric.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
    with open(outputdir + "log_test_micro.txt", "+a") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))
        f.write("{} epoch, {} best eval epoch:Test Loss:{} /Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, best_eval_epoch, test_loss, accuracy,precision,recall,f1))
    confusion, accuracy, precision, recall, f1 = metric.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
    with open(outputdir + "log_test_confusion.txt", "+a") as f:
        for r in range(args.output_dim):
            for c in range(args.output_dim):
                f.write(str(confusion[r][c]))
                if c == args.output_dim -1 :
                    f.write("\n")
                else:
                    f.write("\t")
    with open(outputdir + "log_test_macro.txt", "+a") as f:  
        f.write(time.strftime('%Y-%m-%d %H:%M:%S, ',time.localtime(time.time())))             
        f.write("{} epoch, {} best eval epoch:Test Loss:{} /Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, best_eval_epoch, test_loss, accuracy,precision,recall,f1))

if os.path.isfile(outputdir + "checkpoint.pt"):
    os.remove(outputdir + "checkpoint.pt")

