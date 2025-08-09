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


# ============================ CoNHD_ADMM_Layer =================================
class CoNHD_ADMM_Layer(nn.Module):
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
                 weight_flag=False, ): 
        super(CoNHD_ADMM_Layer, self).__init__()
        self.PE_Block = PE_Block
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.att_type_v = att_type_v
        self.att_type_e = att_type_e
        self.num_att_layer = num_att_layer
        self.dropout = dropout
        self.lnflag = ln
        self.weight_flag = weight_flag
        
        if self.att_type_v == "OrderPE":
            self.pe_v = nn.Linear(weight_dim, input_dim)
        if self.att_type_e == "OrderPE":
            self.pe_e = nn.Linear(weight_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # For Node -> Hyperedge
        self.enc_v = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if PE_Block == 'ISAB': 
                self.enc_v.append(ISAB(input_dim, input_dim, num_heads, num_inds, ln=ln))
            elif PE_Block == 'SAB': 
                self.enc_v.append(SAB(input_dim, input_dim, num_heads, ln=ln)) 
            elif PE_Block == 'UNP': 
                self.enc_v.append(UNP(input_dim, input_dim, input_dim, ln=ln))
        
        # For Hyperedge -> Node
        self.enc_e = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if PE_Block == 'ISAB': 
                self.enc_e.append(ISAB(input_dim, input_dim, num_heads, num_inds, ln=ln))
            elif PE_Block == 'SAB': 
                self.enc_e.append(SAB(input_dim, input_dim, num_heads, ln=ln))
            elif PE_Block == 'UNP': 
                self.enc_e.append(UNP(input_dim, input_dim, input_dim, ln=ln))

        self.update_func = nn.Linear(3*input_dim, input_dim)
        # self.unstruct_diff = MLP(2*input_dim, input_dim, input_dim, 2,
                # dropout=dropout, Normalization='ln' if ln else 'None', InputNorm='ln' if ln else 'None')
        if ln:
            self.ln0 = nn.LayerNorm(input_dim)
        # self.W = MLP(input_dim, input_dim, input_dim, 1, dropout=dropout, Normalization='ln', InputNorm=False)

    def v_message_func(self, edges):
        if self.weight_flag: # weight represents positional information
            return {'v': edges.data['co_feat'], 'msg': edges.data['message'], 'eid': edges.data['_ID'], 'weight': edges.data['weight']}
        else:
            return {'v': edges.data['co_feat'], 'msg': edges.data['message'], 'eid': edges.data['_ID']}
        
    def v_reduce_func(self, nodes): 
        v = nodes.mailbox['v']
        msg = nodes.mailbox['msg']
        eid = nodes.mailbox['eid']
        if self.weight_flag:
            W = nodes.mailbox['weight']
        
        new_msg = 2 * v - msg

        # Attention
        if self.att_type_v == "OrderPE":
            new_msg = new_msg + self.pe_v(W)
        for i, layer in enumerate(self.enc_v):
            new_msg = layer(new_msg)
        new_msg = self.dropout(new_msg)

        new_msg = new_msg + msg - v

        self.reduce_message.append(new_msg.reshape(-1, new_msg.shape[-1]))
        self.reduce_co_eid.append(eid.reshape(-1))
        
        return {}
    
    def e_message_func(self, edges): 
        if self.weight_flag: # weight represents positional information
            return {'v': edges.data['co_feat'], 'msg': edges.data['message'], 'eid': edges.data['_ID'], 'weight': edges.data['weight']}
        else: 
            return {'v': edges.data['co_feat'], 'msg': edges.data['message'], 'eid': edges.data['_ID']}
        
    def e_reduce_func(self, nodes): 
        v = nodes.mailbox['v']
        msg = nodes.mailbox['msg']
        eid = nodes.mailbox['eid']
        if self.weight_flag:
            W = nodes.mailbox['weight']
            
        new_msg = 2 * v - msg

        # Attention
        if self.att_type_e == "OrderPE":
            new_msg = new_msg + self.pe_e(W)
        for i, layer in enumerate(self.enc_e):
            new_msg = layer(new_msg)
        new_msg = self.dropout(new_msg)

        new_msg = new_msg + msg - v
        
        self.reduce_message.append(new_msg.reshape(-1, new_msg.shape[-1]))
        self.reduce_co_eid.append(eid.reshape(-1))
        
        return {}
        
    def order_co_feat(self, co_feat, co_eid, order_co_eid): 
        eid2idx = dict(zip(co_eid.detach().cpu().numpy(), np.arange(co_eid.shape[0])))
        co_idx = list(map(lambda x: eid2idx[x], order_co_eid.detach().cpu().numpy()))
        co_idx = torch.tensor(co_idx, device=co_feat.device)
        return co_feat[co_idx]

    def forward(self, g, next_g, co_feat_in, message_in, co_eid_in, 
                co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0): 
        
        with g.local_scope():
            g.edges['in'].data['co_feat'] = co_feat_in
            g.edges['in'].data['message'] = message_in
            
            self.reduce_message = []
            self.reduce_co_eid = []
            g.update_all(self.v_message_func, self.v_reduce_func, etype='in')
            message_v = torch.concat(self.reduce_message, dim=0)
            co_eid_v = torch.concat(self.reduce_co_eid, dim=0)
            
        with g.local_scope(): 
            g.edges['con'].data['co_feat'] = co_feat_con
            g.edges['con'].data['message'] = message_con
            
            self.reduce_message = []
            self.reduce_co_eid = []
            g.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            message_e = torch.concat(self.reduce_message, dim=0)
            co_eid_e = torch.concat(self.reduce_co_eid, dim=0)
            
        self.reduce_message = []
        self.reduce_co_eid = []

        # organize in edge features for next_g
        co_eid_in = next_g.edges['in'].data['_ID']
        # co_feat_in = self.order_co_feat(co_feat_in, g.edges['in'].data['_ID'], co_eid_in)
        message_v_in = self.order_co_feat(message_v, co_eid_v, co_eid_in)
        message_e_in = self.order_co_feat(message_e, co_eid_e, co_eid_in)
        co_feat_0_in = self.order_co_feat(co_feat_0, co_eid_0, co_eid_in)

        # update co-representation vectors
        co_feat_in = self.update_func(torch.concat([message_v_in, message_e_in, co_feat_0_in], dim=-1))
        # co_feat_in = co_feat_in + co_feat_v_in + co_feat_e_in + self.unstruct_diff(torch.concat([co_feat_in, co_feat_0_in], dim=-1))
        co_feat_in = co_feat_in if getattr(self, 'ln0', None) is None else self.ln0(co_feat_in)
        # co_feat_in = self.W(co_feat_in + co_feat_v_in + co_feat_e_in + co_feat_0_in)

        message_in = message_v_in

        # organize con edge features for next_g
        co_eid_con = next_g.edges['con'].data['_ID']
        # co_feat_con = self.order_co_feat(co_feat_con, g.edges['con'].data['_ID'], co_eid_con)
        message_v_con = self.order_co_feat(message_v, co_eid_v, co_eid_con)
        message_e_con = self.order_co_feat(message_e, co_eid_e, co_eid_con)
        co_feat_0_con = self.order_co_feat(co_feat_0, co_eid_0, co_eid_con)

        # update co-representation vectors
        co_feat_con = self.update_func(torch.concat([message_v_con, message_e_con, co_feat_0_con], dim=-1))
        # co_feat_con = co_feat_con + co_feat_v_con + co_feat_e_con + self.unstruct_diff(torch.concat([co_feat_con, co_feat_0_con], dim=-1))
        co_feat_con = co_feat_con if getattr(self, 'ln0', None) is None else self.ln0(co_feat_con)
        # co_feat_con = self.W(co_feat_con + co_feat_v_con + co_feat_e_con + co_feat_0_con)

        message_con = message_e_con

        co_feat_0 = co_feat_0_in
        co_eid_0 = co_eid_in
            
        return co_feat_in, message_in, co_eid_in, co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0


# ============================ TwoLevelAttNet ===============================
class CoNHD_ADMM(nn.Module): 
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
                 weight_flag=False): 
        super(CoNHD_ADMM, self).__init__()
        self.num_layers = num_layers  # follow the whatsnet, one layer represents information from node to edge to node
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = input_dropout
        if self.input_dropout:
            self.input_dropout = nn.Dropout(input_dropout)

        self.lin_in = torch.nn.Linear(input_vdim, co_rep_dim)

        self.conv = model(PE_Block, co_rep_dim, weight_dim, 
                          att_type_v=att_type_v, att_type_e=att_type_e, num_att_layer=num_att_layer, 
                          num_heads=num_heads, num_inds=num_inds, dropout=dropout, 
                          ln=layernorm, weight_flag=weight_flag)
        
    def forward(self, blocks, vfeat): 
        
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
            co_feat_in = self.input_dropout(co_feat_in)

        # messages
        message_in = co_feat_in
        message_con = co_feat_con

        # initial features for unstructural regularization
        co_feat_0 = co_feat_in
        co_eid_0 = co_eid_in
        
        for i in range(self.num_layers):
            co_feat_in, message_in, co_eid_in, co_feat_con, message_con, co_eid_con, \
                co_feat_0, co_eid_0 = self.conv(blocks[i], blocks[i+1], co_feat_in, message_in, co_eid_in, 
                                                co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0)
            
        # co_feat_in, message_in, co_eid_in, co_feat_con, message_con, co_eid_con, \
        #         co_feat_0, co_eid_0 = self.conv(blocks[-1], blocks[-1], co_feat_in, message_in, co_eid_in, 
        #                                         co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0)
            
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

        # messages
        message_in = co_feat_in
        message_con = co_feat_con

        # initial features for unstructural regularization
        co_feat_0 = co_feat_in
        co_eid_0 = co_eid_in
        
        for i in range(self.num_layers):
            co_feat_in, message_in, co_eid_in, co_feat_con, message_con, co_eid_con, \
                co_feat_0, co_eid_0 = self.conv(blocks[i], blocks[i+1], co_feat_in, message_in, co_eid_in, 
                                                co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0)
            
        # co_feat_in, message_in, co_eid_in, co_feat_con, message_con, co_eid_con, \
        #         co_feat_0, co_eid_0 = self.conv(blocks[-1], blocks[-1], co_feat_in, message_in, co_eid_in, 
        #                                         co_feat_con, message_con, co_eid_con, co_feat_0, co_eid_0)

        return co_feat_in, co_eid_in
