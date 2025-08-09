import os
import numpy as np
from scipy import sparse as sp
from collections import defaultdict

class HyperGraph:
    def __init__(self, inputdir, dataname, exist_hedgename, with_self_loop):
        self.dataname = dataname
        self.hedge2node = []
        self.node2hedge = []
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.node_reindexing = {}
        self.node_orgindex = {}
        self.numhedges = 0
        self.numnodes = 0
        max_size = 0
                        
        filename = 'hypergraph_self_loop.txt' if with_self_loop else 'hypergraph.txt'
        with open(os.path.join(inputdir, dataname, filename), "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                hidx = self.numhedges
                if exist_hedgename:
                    papercode = tmp[0][1:-1] # without '
                    papercode = papercode.rstrip()
                    self.hedgeindex[papercode] = hidx
                    self.hedgename[hidx] = papercode
                    tmp = tmp[1:]
                else:
                    self.hedgeindex[_hidx] = hidx
                    self.hedgename[hidx] = _hidx
                if (max_size < len(tmp)):
                    max_size = len(tmp)
                self.hedge2node.append([])
                for node in tmp:
                    node = node.strip()
                    if node not in self.node_reindexing:
                        node_reindex = self.numnodes
                        self.node_reindexing[node] = node_reindex
                        self.node_orgindex[node_reindex] = node
                        self.node2hedge.append([])
                        self.numnodes += 1
                    nodeindex = self.node_reindexing[node]
                    self.hedge2node[hidx].append(nodeindex)
                    self.node2hedge[nodeindex].append(hidx)
                self.numhedges += 1
         
        print("Max Size = ", max_size)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))

    def construct_weighted_clique(self):
        tmp_dict = defaultdict(int)
        values = []
        rows = []
        cols = []

        for v in range(self.numnodes):
            for h in self.node2hedge[v]:
                for nv in self.hedge2node[h]:
                    if v < nv:
                        key = str(v) + "," + str(nv)
                        tmp_dict[key] += 1 # num of common hedges

        for k, ch in tmp_dict.items():
            v, nv = k.split(",")
            v, nv = int(v), int(nv)
            values.append(ch)
            rows.append(v)
            cols.append(nv)
            
            values.append(ch)
            rows.append(nv)
            cols.append(v)
            
        values = np.array(values)
        rows = np.array(rows)
        cols = np.array(cols)
        self.weighted_matrix = sp.coo_matrix( (values, (rows, cols)), shape=(self.numnodes, self.numnodes))