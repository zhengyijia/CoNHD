import torch
import numpy as np
from collections import defaultdict
import dgl
import os
from tqdm import tqdm, trange
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix

def make_order(ls):
    a = np.array(ls)
    argsorted = np.argsort(a)
    orders = np.zeros(a.shape)
    a = sorted(a)
    
    # adjust
    previous = None
    order = 1
    for i, _a  in enumerate(a):
        if previous is None:
            previous = _a
            orders[argsorted[i]] = order
        elif previous != _a:
            order += 1
            orders[argsorted[i]] = order
            previous = _a
        else:
            orders[argsorted[i]] = order
    return orders.tolist()

class Hypergraph:
    def __init__(self, args):
        self.inputdir = args.data_inputdir + args.task_type + "/"
        self.add_self_loop = args.add_self_loop
        self.task_type = args.task_type
        self.dataset_name = args.dataset_name
        self.exist_hedgename = args.exist_hedgename
        self.valid_inputname = args.valid_inputname
        self.test_inputname = args.test_inputname
        self.use_gpu = args.use_gpu
        self.optim = args.optim
        self.hedge_reg = args.hedge_reg
        self.node_reg = args.node_reg
        self.feat_dim = args.feat_dim
        self.nstep = args.nstep
        
        self.hedge2node = []
        self.node2hedge = [] 
        self.hedge2nodepos = [] # hyperedge index -> node positions
        self.node2hedgePE = []
        self.hedge2nodePE = []
        self.weight_flag = False
        self.hedge2nodeweight = []
        self.node2hedgeweight = []
        self.numhedges = 0
        self.numnodes = 0 
        
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.e_feat = []

        self.node_reindexing = {} # nodeindex -> reindex
        self.node_orgindex = {} # reindex -> nodeindex
        
        self.node2label = []
        
        self.load_graph(args)        
        print("Data is prepared")
        
    def load_graph(self, args):
        # construct connection  -------------------------------------------------------
        self.max_len = 0
        if self.add_self_loop == 'All': 
            filename = 'hypergraph_self_loop.txt' 
        elif self.add_self_loop == 'Isolated': 
            filename = 'hypergraph_isolated_self_loop.txt'
        else: 
            filename = 'hypergraph.txt'
        with open(os.path.join(self.inputdir, self.dataset_name, filename), "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                hidx = self.numhedges
                self.numhedges += 1
                if self.exist_hedgename:
                    papercode = tmp[0][1:-1] # without '
                    papercode = papercode.rstrip()
                    self.hedgeindex[papercode] = hidx
                    self.hedgename[hidx] = papercode
                    tmp = tmp[1:]
                else:
                    self.hedgeindex[_hidx] = hidx
                    self.hedgename[hidx] = _hidx
                # self.hedgeindex[_hidx] = hidx
                # self.hedgename[hidx] = _hidx
                self.hedge2node.append([])
                self.hedge2nodepos.append([])
                self.hedge2nodePE.append([])
                self.hedge2nodeweight.append([])
                self.e_feat.append([])
                if (self.max_len < len(tmp)):
                    self.max_len = len(tmp)
                for node in tmp:
                    node = node.strip()
                    if node not in self.node_reindexing:
                        node_reindex = self.numnodes
                        self.numnodes += 1 
                        self.node_reindexing[node] = node_reindex
                        self.node_orgindex[node_reindex] = node 
                        self.node2hedge.append([])
                        self.node2hedgePE.append([])
                        self.node2hedgeweight.append([])
                    nodeindex = self.node_reindexing[node]
                    self.hedge2node[hidx].append(nodeindex)
                    self.node2hedge[nodeindex].append(hidx)
                    self.hedge2nodePE[hidx].append([])
                    self.node2hedgePE[nodeindex].append([])
                    
        print("Max Size = ", self.max_len)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))
        # update by max degree
        for vhedges in self.node2hedge:
            if self.max_len < len(vhedges):
                self.max_len = len(vhedges)
        for h in range(len(self.e_feat)):
            self.e_feat[h] = [0 for _ in range(args.dim_edge)]
        self.e_feat = torch.tensor(self.e_feat).type('torch.FloatTensor')
        
        # Split Data ------------------------------------------------------------------------
        if self.task_type == 'edge_dependent_node_classification' or self.task_type == 'fit_diffusion' or self.task_type == 'fit_diffusion_from_node_feat': 
            self.test_index = []
            self.valid_index = []
            self.validsize = 0
            self.testsize = 0
            self.trainsize = 0
            self.hedge2type = torch.zeros(self.numhedges)
            
            valid_file_path = os.path.join(self.inputdir, self.dataset_name, self.valid_inputname + ".txt")
            assert os.path.isfile(valid_file_path), "valid file is not exist! "
            with open(valid_file_path, "r") as f:
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    self.valid_index.append(index)
                self.hedge2type[self.valid_index] = 1
                self.validsize = len(self.valid_index)
            test_file_path = os.path.join(self.inputdir, self.dataset_name, self.test_inputname + ".txt")
            if os.path.isfile(test_file_path):
                with open(test_file_path, "r") as f:
                    for line in f.readlines():
                        name = line.rstrip()
                        if self.exist_hedgename is False:
                            name = int(name)
                        index = self.hedgeindex[name]
                        self.test_index.append(index)
                    assert len(self.test_index) > 0
                    self.hedge2type[self.test_index] = 2
                    self.testsize = len(self.test_index)
            self.trainsize = self.numhedges - (self.validsize + self.testsize)
            
        elif self.task_type == 'node_classification' or self.task_type == 'heterophily_experiments': 
            self.node2type = torch.zeros(self.numnodes)
            self.valid_index = torch.LongTensor([self.node_reindexing[origin_index] for origin_index in args.split_idx['valid']])
            self.test_index = torch.LongTensor([self.node_reindexing[origin_index] for origin_index in args.split_idx['test']])
            self.node2type[self.valid_index] = 1
            self.node2type[self.test_index] = 2

        elif self.task_type == 'fit_node_level_diffusion':
            valid_file_path = os.path.join(self.inputdir, self.dataset_name, self.valid_inputname + ".txt")
            assert os.path.isfile(valid_file_path), "valid file is not exist! "
            with open(valid_file_path, "r") as f:
                node_origin_indexes =  f.readlines()
            node_origin_indexes = [node_origin_index.strip() for node_origin_index in node_origin_indexes]
            self.valid_index = torch.LongTensor([self.node_reindexing[origin_index] for origin_index in node_origin_indexes])
            
            test_file_path = os.path.join(self.inputdir, self.dataset_name, self.test_inputname + ".txt")
            assert os.path.isfile(test_file_path), "test file is not exist! "
            with open(test_file_path, "r") as f:
                node_origin_indexes =  f.readlines()
            node_origin_indexes = [node_origin_index.strip() for node_origin_index in node_origin_indexes]
            self.test_index = torch.LongTensor([self.node_reindexing[origin_index] for origin_index in node_origin_indexes])

            self.node2type = torch.zeros(self.numnodes)
            self.node2type[self.valid_index] = 1
            self.node2type[self.test_index] = 2

        elif self.task_type == 'edge_classification': 
            self.test_index = []
            self.valid_index = []
            self.validsize = 0
            self.testsize = 0
            self.trainsize = 0
            self.hedge2type = torch.zeros(self.numhedges)

            valid_file_path = os.path.join(self.inputdir, self.dataset_name, self.valid_inputname + ".txt")
            assert os.path.isfile(valid_file_path), "valid file is not exist! "
            with open(valid_file_path, "r") as f: 
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    self.valid_index.append(index)
                self.hedge2type[self.valid_index] = 1
                self.validsize = len(self.valid_index)

            test_file_path = os.path.join(self.inputdir, self.dataset_name, self.test_inputname + ".txt")
            if os.path.isfile(test_file_path):
                with open(test_file_path, "r") as f:
                    for line in f.readlines():
                        name = line.rstrip()
                        if self.exist_hedgename is False:
                            name = int(name)
                        index = self.hedgeindex[name]
                        self.test_index.append(index)
                    assert len(self.test_index) > 0
                    self.hedge2type[self.test_index] = 2
                    self.testsize = len(self.test_index)
            
            self.trainsize = self.numhedges - (self.validsize + self.testsize)
        
        # extract target ---------------------------------------------------------
        if self.task_type == 'edge_dependent_node_classification': 
            print("Extract edge-dependent node labels", flush=True)
            label_path = os.path.join(self.inputdir, self.dataset_name, "hypergraph_pos.txt")
            with open(label_path, "r") as f:
                for _hidx, line in enumerate(f.readlines()):
                    tmp = line.split("\t")
                    if self.exist_hedgename:
                        papercode = tmp[0][1:-1] # without ''
                        if (papercode not in self.hedgeindex):
                            continue
                        hidx = self.hedgeindex[papercode]
                        tmp = tmp[1:]
                    else:
                        if (_hidx not in self.hedgeindex):
                            continue
                        hidx = self.hedgeindex[_hidx]
                    positions = [int(i) for i in tmp]
                    for nodepos in positions:
                        self.hedge2nodepos[hidx].append(nodepos)
                    
        if self.task_type == 'node_classification' or self.task_type == 'heterophily_experiments': 
            print("Extract node labels", flush=True)
            label_path = os.path.join(self.inputdir, self.dataset_name, "labels.txt")
            with open(label_path, "r") as f: 
                node_labels = f.readlines()
            node_labels = [line.strip().split('\t') for line in node_labels]
            node_labels = [[node_origin_index, int(node_label)] for node_origin_index, node_label in node_labels if node_origin_index in self.node_reindexing]
            node_origin_indexes, node_labels = list(zip(*node_labels))
            node_indexes = [self.node_reindexing[origin_index] for origin_index in node_origin_indexes]
            assert len(node_indexes) == self.numnodes, "labels dimension mismatch numnodes! "
            sort_idx = np.array(node_indexes).argsort()
            self.node2label = np.array(node_labels)[sort_idx]
            self.node2label = torch.tensor(self.node2label).type('torch.LongTensor')

        if self.task_type == 'edge_classification': 
            print("Extract edge labels", flush=True)
            label_path = os.path.join(self.inputdir, self.dataset_name, "labels.txt")
            with open(label_path, "r") as f: 
                edge_labels = f.readlines()
            edge_labels = [line.strip().split('\t') for line in edge_labels]
            edge_labels = [[int(edge_origin_index), int(edge_label)] for edge_origin_index, edge_label in edge_labels if int(edge_origin_index) in self.hedgeindex]
            edge_origin_indexes, edge_labels = list(zip(*edge_labels))
            edge_indexes = [self.hedgeindex[origin_index] for origin_index in edge_origin_indexes]
            assert len(edge_indexes) == self.numhedges, "labels dimension mismatch numhedges! "
            sort_idx = np.array(edge_indexes).argsort()
            self.hedge2label = np.array(edge_labels)[sort_idx]
            self.hedge2label = torch.tensor(self.hedge2label).type('torch.LongTensor')

        if self.task_type == 'fit_diffusion': 
            print("Extract diffusion labels", flush=True)
            origin_feat_path = os.path.join(self.inputdir, self.dataset_name, f'origin_feat_dim_{self.feat_dim}.npy')
            diff_feat_path = os.path.join(self.inputdir, self.dataset_name, f'diff_feat_optim_{self.optim}_hedge_reg_{self.hedge_reg}_node_reg_{self.node_reg}_dim_{self.feat_dim}.npy')
            node_hedge_origin_indexes_path = os.path.join(self.inputdir, self.dataset_name, "node_hedge_origin_indexes.txt")
            with open(node_hedge_origin_indexes_path, 'r') as f: 
                node_hedge_origin_indexes = f.readlines()
            node_origin_indexes, hedge_origin_indexes = tuple(zip(*[line.strip().split('\t') for line in node_hedge_origin_indexes]))
            if not self.exist_hedgename:
                hedge_origin_indexes = tuple(int(hedge_origin_index) for hedge_origin_index in hedge_origin_indexes)
            node_hedge_indexes = list(zip([self.node_reindexing[org_index] for org_index in node_origin_indexes], 
                                          [self.hedgeindex[org_index] for org_index in hedge_origin_indexes]))
            self.node_hedge_indexes_dict = {(node_index, hedge_index): i for i, (node_index, hedge_index) in enumerate(node_hedge_indexes)}
            self.origin_feat = np.load(origin_feat_path)
            self.diff_feat = np.load(diff_feat_path)

        elif self.task_type == 'fit_diffusion_from_node_feat': 
            print("Extract diffusion labels", flush=True)
            diff_feat_path = os.path.join(self.inputdir, self.dataset_name, 
                                          f'diff_feat_optim_{self.optim}_hedge_reg_{self.hedge_reg}_node_reg_{self.node_reg}_nstep_{self.nstep}_dim_{self.feat_dim}.npy')
            node_hedge_origin_indexes_path = os.path.join(self.inputdir, self.dataset_name, 
                                                          f'node_hedge_origin_indexes_optim_{self.optim}_hedge_reg_{self.hedge_reg}_node_reg_{self.node_reg}_nstep_{self.nstep}_dim_{self.feat_dim}.txt')
            with open(node_hedge_origin_indexes_path, 'r') as f: 
                node_hedge_origin_indexes = f.readlines()
            node_origin_indexes, hedge_origin_indexes = tuple(zip(*[line.strip().split('\t') for line in node_hedge_origin_indexes]))
            if not self.exist_hedgename:
                hedge_origin_indexes = tuple(int(hedge_origin_index) for hedge_origin_index in hedge_origin_indexes)
            node_hedge_indexes = list(zip([self.node_reindexing[org_index] for org_index in node_origin_indexes], 
                                          [self.hedgeindex[org_index] for org_index in hedge_origin_indexes]))
            self.node_hedge_indexes_dict = {(node_index, hedge_index): i for i, (node_index, hedge_index) in enumerate(node_hedge_indexes)}
            self.diff_feat = np.load(diff_feat_path)

        elif self.task_type == 'fit_node_level_diffusion': 
            print("Extract node_level diffusion labels", flush=True)
            diff_feat_path = os.path.join(self.inputdir, self.dataset_name, 
                                          f'node_level_diff_feat_optim_{self.optim}_reg_{self.hedge_reg}_nstep_{self.nstep}_dim_{self.feat_dim}.npy')
            node_origin_indexes_path = os.path.join(self.inputdir, self.dataset_name, 
                                                    f"node_origin_indexes_optim_{self.optim}_reg_{self.hedge_reg}_nstep_{self.nstep}_dim_{self.feat_dim}.txt")
            with open(node_origin_indexes_path, 'r') as f: 
                node_origin_indexes = f.readlines()
            node_origin_indexes = [node_origin_index.strip() for node_origin_index in node_origin_indexes]
            node_indexes = [self.node_reindexing[origin_index] for origin_index in node_origin_indexes]
            assert len(node_indexes) == self.numnodes, "labels dimension mismatch numnodes! "
            sort_idx = np.array(node_indexes).argsort()
            self.node2label = np.load(diff_feat_path)[sort_idx]
            self.node2label = torch.tensor(self.node2label).type('torch.FloatTensor')
        
        # extract PE ----------------------------------------------------------------------------------------------------
        # fix
        self.order_dim = 0
        # hedge2nodePE
        if len(args.vorder_input) > 0: # centrality -> PE ------------------------------------------------------------------
            self.order_dim = len(args.vorder_input)
            for inputpath in args.vorder_input:
                vfeat = {} # node -> vfeat
                if self.add_self_loop == 'All':  
                    inputpath = os.path.join(self.inputdir, self.dataset_name, inputpath + '_self_loop' + ".txt")
                elif self.add_self_loop == 'Isolated': 
                    inputpath = os.path.join(self.inputdir, self.dataset_name, inputpath + '_isolated_self_loop' + ".txt")
                else: 
                    inputpath = os.path.join(self.inputdir, self.dataset_name, inputpath + ".txt")
                with open(inputpath, "r") as f:
                    columns = f.readline()
                    columns = columns[:-1].split("\t")
                    for line in f.readlines():
                        line = line.rstrip()
                        tmp = line.split("\t")
                        nodeindex = tmp[0]
                        if nodeindex not in self.node_reindexing:
                            # not include in incidence matrix
                            continue
                        node_reindex = self.node_reindexing[nodeindex]
                        for i, col in enumerate(columns):
                            vfeat[node_reindex] = float(tmp[i])
                for hidx, hedge in enumerate(self.hedge2node):
                    feats = []
                    for v in hedge:
                        feats.append(vfeat[v])
                    orders = make_order(feats)
                    for vorder, v in enumerate(hedge):
                        self.hedge2nodePE[hidx][vorder].append((orders[vorder]) / len(feats))
            # check            
            assert len(self.hedge2nodePE) == self.numhedges
            for hidx in range(self.numhedges):
                assert len(self.hedge2nodePE[hidx]) == len(self.hedge2node[hidx])
                for vorder in self.hedge2nodePE[hidx]:
                    assert len(vorder) == len(args.vorder_input)
            # node2hedgePE
            for vidx, node in enumerate(self.node2hedge):
                orders = []
                for hidx in node:
                    for vorder,_v in enumerate(self.hedge2node[hidx]):
                        if _v == vidx:
                            orders.append(self.hedge2nodePE[hidx][vorder])
                            break
                self.node2hedgePE[vidx] = orders
            # check
            assert len(self.node2hedgePE) == self.numnodes
            for vidx in range(self.numnodes):
                assert len(self.node2hedgePE[vidx]) == len(self.node2hedge[vidx])
                for horder in self.node2hedgePE[vidx]:
                    assert len(horder) == len(args.vorder_input)
            self.weight_flag = True
        
        # For LEGCN ----------------------------------------------------------------------------------
        if args.embedder == "legcn": 
            nodedeg = []
            hedgedeg = []
            for hedges in self.node2hedge:
                nodedeg.append(len(hedges))
            for nodes in self.hedge2node:
                hedgedeg.append(len(nodes))
            self.nodedeg = torch.FloatTensor(nodedeg)
            self.hedgedeg = torch.FloatTensor(hedgedeg)
        # For HNN ----------------------------------------------------------------------------------
        if args.embedder == "hnn":
            print("Extract matrices for HNN")
            nodedeg = []
            hedgedeg = []
            for hedges in self.node2hedge:
                nodedeg.append(len(hedges))
            for nodes in self.hedge2node:
                hedgedeg.append(len(nodes))
            self.invDV = torch.pow(torch.FloatTensor(nodedeg), -1)
            self.invDE = torch.pow(torch.FloatTensor(hedgedeg),-1)
            # calculating PE DE^{-1} := emat, P D^{-1} := vmat
            # P = H DE^{-1} H^{T} D^{-T}, PE = H^{T} D^{-T} H DE^{-1}
            DE = sp.diags([d**(-1) for d in hedgedeg], dtype=np.float32)
            D = sp.diags([d**(-1) for d in nodedeg], dtype=np.float32)
            rows, cols, datas = [], [], []
            for h in range(self.numhedges):
                for vi, w in enumerate(self.hedge2node[h]):
                    v = self.hedge2node[h][vi]
                    rows.append(v)
                    cols.append(h)
                    datas.append(1)
            H = csr_matrix((datas, (rows, cols)), shape=(self.numnodes, self.numhedges), dtype=np.float32)
            print("DE, D, H")
            P = H * DE * H.T * D.T
            PE = H.T * D.T * H * DE
            eMat = PE * DE
            vMat = P * D
            print("eMat, vMat")
            # convert to torch
            rows, cols = eMat.nonzero()
            datas = eMat.data
            self.eMat = torch.sparse_coo_tensor([rows,cols], datas, dtype=torch.float32)
            rows, cols = vMat.nonzero()
            datas = vMat.data
            self.vMat = torch.sparse_coo_tensor([rows,cols], datas, dtype=torch.float32)
            print("torch eMat, vMat")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.invDV = self.invDV.to(device)
            self.invDE = self.invDE.to(device)
            self.eMat = self.eMat.to(device)
            self.vMat = self.vMat.to(device)                
        # For HGNN & HCHA ----------------------------------------------------------------------------------
        if args.embedder == "hgnn" or args.embedder == "hcha":
            nodedeg = []
            hedgedeg = []
            for hedges in self.node2hedge:
                nodedeg.append(len(hedges))
            for nodes in self.hedge2node:
                hedgedeg.append(len(nodes))
            self.DV2 = torch.pow(torch.FloatTensor(nodedeg), -0.5)
            self.invDE = torch.pow(torch.FloatTensor(hedgedeg),-1)
            if self.use_gpu or args.embedder == "hcha":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.DV2 = self.DV2.to(device)
                self.invDE = self.invDE.to(device)
        # applying alpha and beta in HNHN ---------------------------------------------------------
        if args.embedder == "hnhn" or args.embedder == "transformerHNHN":
            print("weight")
            e_weight = []
            v_weight = []
            for neighbor_hedges in self.node2hedge:
                v_weight.append(len(neighbor_hedges))
            for hedge in self.hedge2node:
                e_weight.append(len(hedge))
            use_exp_wt = args.use_exp_wt
            e_reg_weight = torch.zeros(self.numhedges)
            v_reg_weight = torch.zeros(self.numnodes)
            for hidx in range(self.numhedges):
                e_wt = e_weight[hidx]
                e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
                e_reg_weight[hidx] = e_reg_wt
            for vidx in range(self.numnodes):
                v_wt = v_weight[vidx]
                v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
                v_reg_weight[vidx] = v_reg_wt
            v_reg_sum = torch.zeros(self.numnodes) # <- e_reg_weight2v_sum
            e_reg_sum = torch.zeros(self.numhedges) # <- v_reg_weight2e_sum
            for hidx, hedges in enumerate(self.hedge2node):
                for vidx in hedges:
                    v_reg_sum[vidx] += e_reg_wt
                    e_reg_sum[hidx] += v_reg_wt  
            e_reg_sum[e_reg_sum==0] = 1
            v_reg_sum[v_reg_sum==0] = 1
            self.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1)
            self.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1)
            self.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1)
            self.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1)
            # check
            for hidx, hedges in enumerate(self.hedge2node):
                e_reg_sum = self.e_reg_sum[hidx]
                v_reg_sum = 0
                for vidx in hedges:
                    v_reg_sum += self.v_reg_weight[vidx]
                assert abs(e_reg_sum - v_reg_sum) < 1e-4
            if self.use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.e_reg_weight = self.e_reg_weight.to(device)
                self.v_reg_sum = self.v_reg_sum.to(device)
                self.v_reg_weight = self.v_reg_weight.to(device)
                self.e_reg_sum = self.e_reg_sum.to(device)
        # UniGCNII ----------------------------------------------------------------------------------
        if args.embedder == "unigcnii":
            degV = []
            for vidx, hedges in enumerate(self.node2hedge):
                degV.append(len(hedges))
            degE = []
            for eidx, nodes in enumerate(self.hedge2node):
                avgdeg = 0
                for v in nodes:
                    avgdeg += (degV[v] / len(nodes))
                degE.append(avgdeg)
            self.degV = torch.Tensor(degV).pow(-0.5).unsqueeze(-1)
            self.degE = torch.Tensor(degE).pow(-0.5).unsqueeze(-1)
            if self.use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.degV = self.degV.to(device)
                self.degE = self.degE.to(device)

    def get_data(self, type=0):
        if self.task_type == 'edge_dependent_node_classification' or self.task_type == 'fit_diffusion' or self.task_type == 'fit_diffusion_from_node_feat': 
            hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
            if self.use_gpu is False:
                hedgelist = hedgelist.tolist()
            return hedgelist
        if self.task_type == 'node_classification' or self.task_type == 'heterophily_experiments': 
            nodelist = ((self.node2type == type).nonzero(as_tuple=True)[0])
            if self.use_gpu is False: 
                nodelist = nodelist.tolist()
            return nodelist
        if self.task_type == 'edge_classification': 
            hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
            if self.use_gpu is False:
                hedgelist = hedgelist.tolist()
            return hedgelist
        if self.task_type == 'fit_node_level_diffusion': 
            nodelist = ((self.node2type == type).nonzero(as_tuple=True)[0])
            if self.use_gpu is False: 
                nodelist = nodelist.tolist()
            return nodelist
    
# Generate DGL Graph ==============================================================================================
def gen_DGLGraph(hedge2node, hedge2nodepos):
    data_dict = defaultdict(list)
    in_edge_label = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_label.append(hedge2nodepos[hidx][vorder]) 
            con_edge_label.append(hedge2nodepos[hidx][vorder]) 

    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    return g

def gen_weighted_DGLGraph(hedge2node, hedge2nodePE, hedge2nodepos):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_weights = []
    in_edge_label = []
    con_edge_weights = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)
            # label
            in_edge_label.append(hedge2nodepos[hidx][vorder])
            con_edge_label.append(hedge2nodepos[hidx][vorder])

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    
    return g

# Generate DGL Graph without labels ==============================================================================================
def gen_DGLGraph_without_label(hedge2node):
    data_dict = defaultdict(list)
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))

    g = dgl.heterograph(data_dict)
    
    return g

def gen_weighted_DGLGraph_without_label(hedge2node, hedge2nodePE):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_weights = []
    con_edge_weights = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    
    return g

# Generate DGL Graph with diff feat ==============================================================================================
def gen_DGLGraph_with_diff_feat(hedge2node, 
                                node_hedge_indexes_dict, diff_feat): 
    data_dict = defaultdict(list)
    # in_edge_origin_feat = []
    # con_edge_origin_feat = []
    in_edge_diff_feat = []
    con_edge_diff_feat = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # in_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]]) 
            # con_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]])
            in_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])
            con_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])

    # in_edge_origin_feat = torch.Tensor(np.array(in_edge_origin_feat))
    # con_edge_origin_feat = torch.Tensor(np.array(con_edge_origin_feat))
    in_edge_diff_feat = torch.Tensor(np.array(in_edge_diff_feat))
    con_edge_diff_feat = torch.Tensor(np.array(con_edge_diff_feat))

    g = dgl.heterograph(data_dict)
    # g['in'].edata['origin_feat'] = in_edge_origin_feat
    # g['con'].edata['origin_feat'] = con_edge_origin_feat
    g['in'].edata['diff_feat'] = in_edge_diff_feat
    g['con'].edata['diff_feat'] = con_edge_diff_feat
    return g

def gen_weighted_DGLGraph_with_diff_feat(hedge2node, hedge2nodePE, 
                                         node_hedge_indexes_dict, diff_feat):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    # in_edge_origin_feat = []
    # con_edge_origin_feat = []
    in_edge_diff_feat = []
    con_edge_diff_feat = []
    in_edge_weights = []
    con_edge_weights = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)
            # diff feat
            # in_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]]) 
            # con_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]])
            in_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])
            con_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    # in_edge_origin_feat = torch.Tensor(in_edge_origin_feat)
    # con_edge_origin_feat = torch.Tensor(con_edge_origin_feat)
    in_edge_diff_feat = torch.Tensor(in_edge_diff_feat)
    con_edge_diff_feat = torch.Tensor(con_edge_diff_feat)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    # g['in'].edata['origin_feat'] = in_edge_origin_feat
    # g['con'].edata['origin_feat'] = con_edge_origin_feat
    g['in'].edata['diff_feat'] = in_edge_diff_feat
    g['con'].edata['diff_feat'] = con_edge_diff_feat
    
    return g

# Generate DGL Graph with diff feat ==============================================================================================
def gen_DGLGraph_with_origin_feat_and_diff_feat(hedge2node, node_hedge_indexes_dict, 
                                                origin_feat, diff_feat): 
    data_dict = defaultdict(list)
    in_edge_origin_feat = []
    con_edge_origin_feat = []
    in_edge_diff_feat = []
    con_edge_diff_feat = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]]) 
            con_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]])
            in_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])
            con_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])

    in_edge_origin_feat = torch.Tensor(np.array(in_edge_origin_feat))
    con_edge_origin_feat = torch.Tensor(np.array(con_edge_origin_feat))
    in_edge_diff_feat = torch.Tensor(np.array(in_edge_diff_feat))
    con_edge_diff_feat = torch.Tensor(np.array(con_edge_diff_feat))

    g = dgl.heterograph(data_dict)
    g['in'].edata['origin_feat'] = in_edge_origin_feat
    g['con'].edata['origin_feat'] = con_edge_origin_feat
    g['in'].edata['diff_feat'] = in_edge_diff_feat
    g['con'].edata['diff_feat'] = con_edge_diff_feat
    return g

def gen_weighted_DGLGraph_with_origin_feat_and_diff_feat(hedge2node, hedge2nodePE, node_hedge_indexes_dict, 
                                                         origin_feat, diff_feat):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_origin_feat = []
    con_edge_origin_feat = []
    in_edge_diff_feat = []
    con_edge_diff_feat = []
    in_edge_weights = []
    con_edge_weights = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)
            # diff feat
            in_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]]) 
            con_edge_origin_feat.append(origin_feat[node_hedge_indexes_dict[(v, hidx)]])
            in_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])
            con_edge_diff_feat.append(diff_feat[node_hedge_indexes_dict[(v, hidx)]])

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    in_edge_origin_feat = torch.Tensor(in_edge_origin_feat)
    con_edge_origin_feat = torch.Tensor(con_edge_origin_feat)
    in_edge_diff_feat = torch.Tensor(in_edge_diff_feat)
    con_edge_diff_feat = torch.Tensor(con_edge_diff_feat)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    g['in'].edata['origin_feat'] = in_edge_origin_feat
    g['con'].edata['origin_feat'] = con_edge_origin_feat
    g['in'].edata['diff_feat'] = in_edge_diff_feat
    g['con'].edata['diff_feat'] = con_edge_diff_feat
    
    return g
