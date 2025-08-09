import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
import time
import pickle
from scipy.sparse import coo_matrix
import dgl
import dgl.function as fn
from model.layer import FC, MLP

# Based on https://github.com/Graph-COM/ED-HNN
# UNweighted Pooling permutation equivariant block
class UNP(nn.Module):
    # use for ablation study of positional encodings
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0, mlp1_layers=2, mlp2_layers=2, ln=False):
        super(UNP, self).__init__()
        if mlp1_layers > 0:
            self.W1 = MLP(dim_in, dim_hidden, dim_out, mlp1_layers,
                dropout=dropout, Normalization='ln' if ln else 'None', InputNorm='ln' if ln else 'None')
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(dim_in+dim_out, dim_hidden, dim_out, mlp2_layers,
                dropout=dropout, Normalization='ln' if ln else 'None', InputNorm='ln' if ln else 'None')
        else:
            self.W2 = lambda X: X[..., dim_in:]
    def forward(self, X):        
        X_mean = self.W1(X).mean(dim=1).unsqueeze(1)
        X_mean = X_mean.repeat(1, X.shape[1], 1)
        out = self.W2(torch.cat([X, X_mean], dim=-1))
        
        return out

# Based on SetTransformer https://github.com/juho-lee/set_transformer

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, numlayers=1):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.numlayers = numlayers
        self.fc_o = nn.ModuleList()
        for _ in range(numlayers):
            self.fc_o.append(nn.Linear(dim_V, dim_V))
        
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(torch.matmul(Q_, torch.transpose(K_, 1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + torch.matmul(A, V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        resO = O
        for i, lin in enumerate(self.fc_o[:-1]):
            O = F.relu(lin(O), inplace=True)
        O = resO + F.relu(self.fc_o[-1](O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    # use for ablation study of positional encodings
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def forward(self, X):
        out = self.mab(X, X)
        return out

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        # X is dim_in, I is (num_inds) * (dim_out)
        # After mab0, I is represented by X => H = (num_inds) * (dim_out)
        # After mab1, X is represented by H => X' = (X.size[1]) * (dim_in)
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_seeds, ln=False, numlayers=1):
        # (num_seeds, dim) is represented by X
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_out))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln,  numlayers=numlayers)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# ============================ CoNHD_GD_Layer =================================
class CoNHD_GD_Layer(nn.Module):
    def __init__(self, 
                 PE_Block, 
                 input_dim, 
                 weight_dim,
                 att_type_v = "OrderPE",
                 att_type_e = "OrderPE",
                 num_att_layer = 2,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 weight_flag=False, 
                 node_agg=False,  # for ablation
                 hedge_agg=False,  # for ablation
                 ): 
        super(CoNHD_GD_Layer, self).__init__()
        self.PE_Block = PE_Block
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.att_type_v = att_type_v
        self.att_type_e = att_type_e
        self.num_att_layer = num_att_layer
        self.dropout = dropout
        self.lnflag = ln
        self.weight_flag = weight_flag
        self.node_agg = node_agg
        self.hedge_agg = hedge_agg
        
        if self.att_type_v == "OrderPE":
            self.pe_v = nn.Linear(weight_dim, input_dim)
        if self.att_type_e == "OrderPE":
            self.pe_e = nn.Linear(weight_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # For Node -> Hyperedge
        dimension = input_dim
        self.enc_v = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if PE_Block == 'ISAB': 
                self.enc_v.append(ISAB(dimension, dimension, num_heads, num_inds, ln=ln))
            elif PE_Block == 'SAB': 
                self.enc_v.append(SAB(dimension, dimension, num_heads, ln=ln)) 
            elif PE_Block == 'UNP': 
                self.enc_v.append(UNP(dimension, dimension, dimension, ln=ln))
        
        # For Hyperedge -> Node
        dimension = input_dim
        self.enc_e = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if PE_Block == 'ISAB': 
                self.enc_e.append(ISAB(dimension, dimension, num_heads, num_inds, ln=ln))
            elif PE_Block == 'SAB': 
                self.enc_e.append(SAB(dimension, dimension, num_heads, ln=ln))
            elif PE_Block == 'UNP': 
                self.enc_e.append(UNP(dimension, dimension, dimension, ln=ln))

        self.update_func = nn.Linear(4*input_dim, input_dim)
        # self.unstruct_diff = nn.Linear(2*input_dim, input_dim)
        # self.unstruct_diff = MLP(2*input_dim, input_dim, input_dim, 2,
                # dropout=dropout, Normalization='ln' if ln else 'None', InputNorm='ln' if ln else 'None')
        if ln:
            self.ln0 = nn.LayerNorm(input_dim)
        # self.W = MLP(input_dim, input_dim, input_dim, 1, dropout=dropout, Normalization='ln', InputNorm=False)

    def v_message_func(self, edges):
        if self.weight_flag: # weight represents positional information
            return {'v': edges.data['co_feat'], 'eid': edges.data['_ID'], 'weight': edges.data['weight']}
        else:
            return {'v': edges.data['co_feat'], 'eid': edges.data['_ID']}
        
    def v_reduce_func(self, nodes): 
        v = nodes.mailbox['v']
        eid = nodes.mailbox['eid']
        if self.weight_flag:
            W = nodes.mailbox['weight']
        
        # Attention
        if self.att_type_v == "OrderPE":
            v = v + self.pe_v(W)
        for i, layer in enumerate(self.enc_v):
            v = layer(v)
        v = self.dropout(v)

        # Mean aggregation, for ablation
        if self.node_agg: 
            v = v.mean(1).unsqueeze(1).repeat(1, v.shape[1], 1)

        self.reduce_co_feat.append(v.reshape(-1, v.shape[-1]))
        self.reduce_co_eid.append(eid.reshape(-1))
        
        return {}
    
    def e_message_func(self, edges): 
        if self.weight_flag: # weight represents positional information
            return {'v': edges.data['co_feat'], 'eid': edges.data['_ID'], 'weight': edges.data['weight']}
        else: 
            return {'v': edges.data['co_feat'], 'eid': edges.data['_ID']}
        
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']
        eid = nodes.mailbox['eid']
        if self.weight_flag:
            W = nodes.mailbox['weight']
            
        # Attention
        if self.att_type_e == "OrderPE":
            v = v + self.pe_e(W)
        for i, layer in enumerate(self.enc_e):
            v = layer(v)
        v = self.dropout(v)
        
        # Mean aggregation, for ablation
        if self.hedge_agg: 
            v = v.mean(1).unsqueeze(1).repeat(1, v.shape[1], 1)

        self.reduce_co_feat.append(v.reshape(-1, v.shape[-1]))
        self.reduce_co_eid.append(eid.reshape(-1))
        
        return {}
        
    def order_co_feat(self, co_feat, co_eid, order_co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))
        co_idx = list(map(lambda x: eid2idx[x], order_co_eid.detach().cpu().numpy()))
        co_idx = torch.tensor(co_idx, device=co_feat.device)
        return co_feat[co_idx]

    def forward(self, g, next_g, co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0): 
        
        with g.local_scope():
            g.edges['in'].data['co_feat'] = co_feat_in
            
            self.reduce_co_feat = []
            self.reduce_co_eid = []
            g.update_all(self.v_message_func, self.v_reduce_func, etype='in')
            co_feat_v = torch.concat(self.reduce_co_feat, dim=0)
            co_eid_v = torch.concat(self.reduce_co_eid, dim=0)
            
        with g.local_scope(): 
            g.edges['con'].data['co_feat'] = co_feat_con
            
            self.reduce_co_feat = []
            self.reduce_co_eid = []
            g.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            co_feat_e = torch.concat(self.reduce_co_feat, dim=0)
            co_eid_e = torch.concat(self.reduce_co_eid, dim=0)
            
        self.reduce_co_feat = []
        self.reduce_co_eid = []

        # organize in edge features for next_g
        co_eid_in = next_g.edges['in'].data['_ID']
        co_feat_in = self.order_co_feat(co_feat_in, g.edges['in'].data['_ID'], co_eid_in)
        co_feat_v_in = self.order_co_feat(co_feat_v, co_eid_v, co_eid_in)
        co_feat_e_in = self.order_co_feat(co_feat_e, co_eid_e, co_eid_in)
        co_feat_0_in = self.order_co_feat(co_feat_0, co_eid_0, co_eid_in)

        # update co-representation vectors
        co_feat_in = self.update_func(torch.concat([co_feat_in, co_feat_v_in, co_feat_e_in, co_feat_0_in], dim=-1))
        # co_feat_in = co_feat_in + co_feat_v_in + co_feat_e_in + self.unstruct_diff(torch.concat([co_feat_in, co_feat_0_in], dim=-1))
        co_feat_in = co_feat_in if getattr(self, 'ln0', None) is None else self.ln0(co_feat_in)

        # organize con edge features for next_g
        co_eid_con = next_g.edges['con'].data['_ID']
        co_feat_con = self.order_co_feat(co_feat_con, g.edges['con'].data['_ID'], co_eid_con)
        co_feat_v_con = self.order_co_feat(co_feat_v, co_eid_v, co_eid_con)
        co_feat_e_con = self.order_co_feat(co_feat_e, co_eid_e, co_eid_con)
        co_feat_0_con = self.order_co_feat(co_feat_0, co_eid_0, co_eid_con)

        # update co-representation vectors
        co_feat_con = self.update_func(torch.concat([co_feat_con, co_feat_v_con, co_feat_e_con, co_feat_0_con], dim=-1))
        # co_feat_con = co_feat_con + co_feat_v_con + co_feat_e_con + self.unstruct_diff(torch.concat([co_feat_con, co_feat_0_con], dim=-1))
        co_feat_con = co_feat_con if getattr(self, 'ln0', None) is None else self.ln0(co_feat_con)

        co_feat_0 = co_feat_0_in
        co_eid_0 = co_eid_in
            
        return co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0


# ============================ CoNHD ===============================
class CoNHD_GD(nn.Module): 
    def __init__(self, 
                 model, 
                 PE_Block, 
                 input_vdim, 
                 co_rep_dim, 
                 weight_dim=0,
                 num_layers=2,
                 num_heads=4,
                 num_inds=4,
                 att_type_v = "OrderPE",
                 att_type_e = "OrderPE",
                 num_att_layer = 2,
                 layernorm = False,
                 dropout=0.6,
                 input_dropout=0, 
                 input_vfeat_dropout=0, 
                 weight_flag=False, 
                 node_agg=False,  # for ablation
                 hedge_agg=False,  # for ablation
                 input_edim=None, 
                 with_edge_feat=False, 
                 share_weights=True):
        super(CoNHD_GD, self).__init__()
        self.num_layers = num_layers  # follow the whatsnet, one layer represents information from node to edge to node
        self.input_dropout = input_dropout
        self.input_vfeat_dropout = input_vfeat_dropout
        self.dropout = nn.Dropout(dropout)
        if self.input_dropout:
            self.input_dropout = nn.Dropout(input_dropout)
        if self.input_vfeat_dropout:
            self.input_vfeat_dropout = nn.Dropout(input_vfeat_dropout)
        

        if not with_edge_feat:
            self.lin_in = torch.nn.Linear(input_vdim, co_rep_dim)
        else: 
            self.lin_in = torch.nn.Linear(input_vdim+input_edim, co_rep_dim)

        self.share_weights = share_weights
        if self.share_weights:
            self.conv = model(PE_Block, co_rep_dim, weight_dim, 
                            att_type_v=att_type_v, att_type_e=att_type_e, num_att_layer=num_att_layer, 
                            num_heads=num_heads, num_inds=num_inds, dropout=dropout, 
                            ln=layernorm, weight_flag=weight_flag, node_agg=node_agg, hedge_agg=hedge_agg)
        else:
            self.conv = nn.ModuleList()
            for i in range(self.num_layers):
                self.conv.append(model(PE_Block, co_rep_dim, weight_dim, 
                            att_type_v=att_type_v, att_type_e=att_type_e, num_att_layer=num_att_layer, 
                            num_heads=num_heads, num_inds=num_inds, dropout=dropout, 
                            ln=layernorm, weight_flag=weight_flag, node_agg=node_agg, hedge_agg=hedge_agg))
        
    def forward(self, blocks, vfeat): 
        
        if self.input_vfeat_dropout: 
            vfeat = self.input_vfeat_dropout(vfeat)
        
        # organize in edges features
        co_eid_in = blocks[0].edges['in'].data['_ID']
        in_src, in_dst = blocks[0].edges(etype='in')
        co_feat_in = vfeat[in_src]
        co_feat_in = F.relu(self.lin_in(co_feat_in))
        if self.input_dropout: 
            co_feat_in = self.input_dropout(co_feat_in)

        # organize con edges features
        co_eid_con = blocks[0].edges['con'].data['_ID']
        con_src, con_dst = blocks[0].edges(etype='con')
        co_feat_con = vfeat[con_dst]
        co_feat_con = F.relu(self.lin_in(co_feat_con))
        if self.input_dropout: 
            co_feat_con = self.input_dropout(co_feat_con)

        # initial features for unstructural regularization
        co_feat_0 = co_feat_in
        co_eid_0 = co_eid_in
        
        for i in range(self.num_layers):
            if self.share_weights:
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv(blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)
            else: 
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv[i](blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)
        
        return co_feat_in, co_eid_in
    
    def forward_with_edge_feat(self, blocks, vfeat, efeat): 
        # organize in edges features
        co_eid_in = blocks[0].edges['in'].data['_ID']
        in_src, in_dst = blocks[0].edges(etype='in')
        co_feat_in = torch.concat([vfeat[in_src], efeat[in_dst]], dim=-1)
        co_feat_in = F.relu(self.lin_in(co_feat_in))
        if self.input_dropout: 
            co_feat_in = self.dropout(co_feat_in)
            
        # organize con edges features
        co_eid_con = blocks[0].edges['con'].data['_ID']
        con_src, con_dst = blocks[0].edges(etype='con')
        co_feat_con = torch.concat([vfeat[con_dst], efeat[con_src]], dim=-1)
        co_feat_con = F.relu(self.lin_in(co_feat_con))
        if self.input_dropout: 
            co_feat_con = self.dropout(co_feat_con)
    
        # initial features for unstructural regularization
        co_feat_0 = co_feat_in
        co_eid_0 = co_eid_in
        
        for i in range(self.num_layers):
            if self.share_weights:
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv(blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)
            else: 
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv[i](blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)

        return co_feat_in, co_eid_in
    
    def fit_diffusion(self, blocks): 
        # organize in edges features
        co_eid_in = blocks[0].edges['in'].data['_ID']
        co_feat_in = blocks[0].edges['in'].data['origin_feat']
        co_feat_in = F.relu(self.lin_in(co_feat_in))
        if self.input_dropout: 
            co_feat_in = self.dropout(co_feat_in)

        # organize con edges features
        co_eid_con = blocks[0].edges['con'].data['_ID']
        co_feat_con = blocks[0].edges['con'].data['origin_feat']
        co_feat_con = F.relu(self.lin_in(co_feat_con))
        if self.input_dropout: 
            co_feat_con = self.dropout(co_feat_con)

        # initial features for unstructural regularization
        co_feat_0 = co_feat_in
        co_eid_0 = co_eid_in
        
        for i in range(self.num_layers):
            if self.share_weights:
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv(blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)
            else: 
                co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0 = self.conv[i](blocks[i], blocks[i+1], co_feat_in, co_eid_in, co_feat_con, co_eid_con, co_feat_0, co_eid_0)
        
        return co_feat_in, co_eid_in


# ============================ CoNHDScorer ===============================
class CoNHDScorer(nn.Module): 
    def __init__(self, 
                 num_classes, 
                 co_rep_dim, 
                 dim_hidden = 128, 
                 num_layers=1): 
        super(CoNHDScorer, self).__init__()
        self.co_rep_dim, self.num_classes, self.dim_hidden = co_rep_dim, num_classes, dim_hidden
        self.num_layers = num_layers
        
        self.predict_layer = FC(co_rep_dim, dim_hidden, num_classes, num_layers)
        
    def message_func(self, edges): 
        return {'v': edges.data['co_feat'], 'label': edges.data['label'].long(), 'src': edges.src['_ID'], 'dst': edges.dst['_ID']}
    
    def reduce_func(self, nodes): 
        L = nodes.mailbox['label']
        v = nodes.mailbox['v']
        src = nodes.mailbox['src']
        dst = nodes.mailbox['dst']
        
        v = self.predict_layer(v)

        self.output.append(v.reshape(-1, self.num_classes))
        self.labels.append(L.reshape(-1))
        self.node_indexes.append(src.reshape(-1))
        self.hedge_indexes.append(dst.reshape(-1))

        return {}

    def forward(self, g, co_feat, co_eid): 
        self.output = []
        self.labels = []
        self.node_indexes = []
        self.hedge_indexes = []
        
        with g.local_scope(): 
            eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))
            co_idx = list(map(lambda x: eid2idx[x], g.edges['in'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['in'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.message_func, self.reduce_func, etype='in')
        
        output = torch.cat(self.output, dim=0)
        labels = torch.cat(self.labels, dim=0)
        node_indexes = torch.cat(self.node_indexes, dim=0)
        hedge_indexes = torch.cat(self.hedge_indexes, dim=0)

        return output, labels, node_indexes, hedge_indexes
    

# ============================ CoNHDDiffScorer ===============================
class CoNHDDiffScorer(nn.Module): 
    def __init__(self, 
                 output_dim, 
                 co_rep_dim, 
                 dim_hidden = 128, 
                 num_layers=1): 
        super(CoNHDDiffScorer, self).__init__()
        self.co_rep_dim, self.output_dim, self.dim_hidden = co_rep_dim, output_dim, dim_hidden
        self.num_layers = num_layers
        
        self.predict_layer = FC(co_rep_dim, dim_hidden, output_dim, num_layers)
        
    def message_func(self, edges): 
        return {'v': edges.data['co_feat'], 'label': edges.data['diff_feat'], 'src': edges.src['_ID'], 'dst': edges.dst['_ID']}
    
    def reduce_func(self, nodes): 
        L = nodes.mailbox['label']
        v = nodes.mailbox['v']
        src = nodes.mailbox['src']
        dst = nodes.mailbox['dst']
        
        v = self.predict_layer(v)

        self.output.append(v.reshape(-1, self.output_dim))
        self.labels.append(L.reshape(-1, self.output_dim))
        self.node_indexes.append(src.reshape(-1))
        self.hedge_indexes.append(dst.reshape(-1))

        return {}

    def forward(self, g, co_feat, co_eid): 
        self.output = []
        self.labels = []
        self.node_indexes = []
        self.hedge_indexes = []
        
        with g.local_scope(): 
            eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))
            co_idx = list(map(lambda x: eid2idx[x], g.edges['in'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['in'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.message_func, self.reduce_func, etype='in')
        
        output = torch.cat(self.output, dim=0)
        labels = torch.cat(self.labels, dim=0)
        node_indexes = torch.cat(self.node_indexes, dim=0)
        hedge_indexes = torch.cat(self.hedge_indexes, dim=0)

        return output, labels, node_indexes, hedge_indexes


class CoNHDNodeScorer(nn.Module): 
    def __init__(self, 
                 num_classes, 
                 co_rep_dim, 
                 hidden_dim, 
                 dropout=0.6,
                 num_layer=1, 
                 layernorm=False): 
        super(CoNHDNodeScorer, self).__init__()
        self.co_rep_dim = co_rep_dim
        self.num_classes = num_classes
        self.num_layer = num_layer

        self.dropout = nn.Dropout(dropout)
        self.classifier = MLP(in_channels=co_rep_dim,
                              hidden_channels=hidden_dim,
                              out_channels=num_classes,
                              num_layers=num_layer,
                              dropout=dropout,
                              Normalization='ln' if layernorm else 'None',
                              InputNorm=False)
        
    def e_message_func(self, edges): 
        return {'v': edges.data['co_feat']}
    
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']

        # Aggregate
        o = v.mean(dim=1)
                
        return {'o': o}

    def forward(self, g, co_feat, co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))

        with g.local_scope():
            co_idx = list(map(lambda x: eid2idx[x], g.edges['con'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['con'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            vfeat = g.dstnodes['node'].data['o']
            vfeat = vfeat.squeeze(1)

        vfeat = self.dropout(vfeat)
        predictions = self.classifier(vfeat)

        return predictions
    

class CoNHDAttNodeScorer(nn.Module): 
    def __init__(self, 
                 num_classes, 
                 co_rep_dim, 
                 hidden_dim, 
                 dropout=0.6,
                 num_layer=1, 
                 layernorm=False): 
        super(CoNHDAttNodeScorer, self).__init__()
        self.co_rep_dim = co_rep_dim
        self.num_classes = num_classes
        self.num_layer = num_layer

        self.dropout = nn.Dropout(dropout)
        self.I = nn.Parameter(torch.Tensor(1, 1, co_rep_dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(co_rep_dim, co_rep_dim, co_rep_dim, num_heads=4, ln=layernorm, numlayers=2)
        self.classifier = MLP(in_channels=co_rep_dim,
                              hidden_channels=hidden_dim,
                              out_channels=num_classes,
                              num_layers=num_layer,
                              dropout=dropout,
                              Normalization='ln' if layernorm else 'None',
                              InputNorm=True)
        
    def e_message_func(self, edges): 
        return {'v': edges.data['co_feat']}
    
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']

        # Aggregate
        o = self.mab(self.I.repeat(v.size(0), 1, 1), v).squeeze(1)
        # o = v.mean(dim=1)
                
        return {'o': o}

    def forward(self, g, co_feat, co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))

        with g.local_scope():
            co_idx = list(map(lambda x: eid2idx[x], g.edges['con'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['con'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            vfeat = g.dstnodes['node'].data['o']
            vfeat = vfeat.squeeze(1)

        vfeat = self.dropout(vfeat)
        predictions = self.classifier(vfeat)

        return predictions


class CoNHDMeanNodeScorer(nn.Module): 
    def __init__(self, 
                 num_classes, 
                 co_rep_dim, 
                 hidden_dim, 
                 dropout=0.6,
                 num_layer=1, 
                 layernorm=False): 
        super(CoNHDMeanNodeScorer, self).__init__()
        self.co_rep_dim = co_rep_dim
        self.num_classes = num_classes
        self.num_layer = num_layer

        self.dropout = nn.Dropout(dropout)
        self.classifier = MLP(in_channels=co_rep_dim,
                              hidden_channels=hidden_dim,
                              out_channels=num_classes,
                              num_layers=num_layer,
                              dropout=dropout,
                              Normalization='ln' if layernorm else 'None',
                              InputNorm=False)
        
    def e_message_func(self, edges): 
        return {'v': edges.data['co_feat']}
    
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']

        v = self.dropout(v)
        o = self.classifier(v)

        # Aggregate
        # mean
        # o = v.mean(dim=1)
        # max
        o = o[range(o.shape[0]), F.softmax(o, dim=-1).max(dim=-1)[0].argmax(dim=-1)]
                
        return {'o': o}

    def forward(self, g, co_feat, co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))

        with g.local_scope():
            co_idx = list(map(lambda x: eid2idx[x], g.edges['con'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['con'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            predictions = g.dstnodes['node'].data['o']
            predictions = predictions.squeeze(1)

        return predictions
    
class CoNHDEdgeScorer(nn.Module): 
    def __init__(self, 
                 num_classes, 
                 co_rep_dim, 
                 hidden_dim, 
                 dropout=0.6,
                 num_layer=1, 
                 layernorm=False): 
        super(CoNHDEdgeScorer, self).__init__()
        self.co_rep_dim = co_rep_dim
        self.num_classes = num_classes
        self.num_layer = num_layer

        self.dropout = nn.Dropout(dropout)
        self.classifier = MLP(in_channels=co_rep_dim,
                              hidden_channels=hidden_dim,
                              out_channels=num_classes,
                              num_layers=num_layer,
                              dropout=dropout,
                              Normalization='ln' if layernorm else 'None',
                              InputNorm=False)
        
    def e_message_func(self, edges): 
        return {'v': edges.data['co_feat']}
    
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']

        # Aggregate
        o = v.mean(dim=1)
                
        return {'o': o}

    def forward(self, g, co_feat, co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))

        with g.local_scope():
            co_idx = list(map(lambda x: eid2idx[x], g.edges['in'].data['_ID'].detach().cpu().numpy()))
            co_idx = torch.tensor(co_idx, device=g.device)
            g.edges['in'].data['co_feat'] = co_feat[co_idx]
            g.update_all(self.e_message_func, self.e_reduce_func, etype='in')
            efeat = g.dstnodes['edge'].data['o']
            efeat = efeat.squeeze(1)

        efeat = self.dropout(efeat)
        predictions = self.classifier(efeat)

        return predictions
