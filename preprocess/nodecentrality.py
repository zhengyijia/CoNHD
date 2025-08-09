import argparse
import networkx as nx
import os

from data_load import HyperGraph

def cal_kcore(graph):
    node_centrality = {}
    pos = [-1 for _ in range(graph.numnodes)]
    vert = [-1 for _ in range(graph.numnodes)]
    check_hedge = [False for _ in range(graph.numhedges)]

    node_degree = []
    md = 0
    for vidx in range(graph.numnodes):
        deg = len(graph.node2hedge[vidx])
        node_degree.append(deg)
        node_centrality[vidx] = deg
        if md < deg:
            md = deg
    bin = [0 for _ in range(md + 1)]
    for v in range(graph.numnodes):
        vdeg = node_degree[v]
        bin[vdeg] += 1

    start = 0
    for d in range(md + 1):
        num = bin[d]
        bin[d] = start
        start += num

    for v in range(graph.numnodes):
        pos[v] = bin[node_degree[v]]
        vert[pos[v]] = v
        bin[node_degree[v]] += 1

    for d in range(md, 0, -1):
        bin[d] = bin[d-1]
    bin[0] = 0

    previous = -1
    for i in range(graph.numnodes):
        v = vert[i]
        assert previous <= node_centrality[v]
        previous = node_centrality[v]

    for i in range(graph.numnodes):
        v = vert[i]
        vdeg = node_degree[v]
        for hidx in range(vdeg):
            h = graph.node2hedge[v][hidx]
            if check_hedge[h] is False:
                hsize = len(graph.hedge2node[h])
                for nvidx in range(hsize):
                    nv = graph.hedge2node[h][nvidx]
                    if node_centrality[nv] > node_centrality[v]:
                        dnv = node_centrality[nv]
                        pnv = pos[nv]
                        pw = bin[dnv]
                        w = vert[pw]
                        if nv != w:
                            pos[nv] = pw
                            pos[w] = pnv
                            vert[pnv] = w
                            vert[pw] = nv
                        bin[dnv] += 1
                        node_centrality[nv] -= 1
                check_hedge[h] = True

    return node_centrality

def cal_degree(graph):
    node_centrality = {}
    for vidx in range(graph.numnodes):
        deg = len(graph.node2hedge[vidx])
        node_centrality[vidx] = deg
    return node_centrality

def cal_pagerank(graph):
    graph.construct_weighted_clique()
    nx_graph = nx.convert_matrix.from_scipy_sparse_matrix(graph.weighted_matrix)

    return nx.algorithms.link_analysis.pagerank_alg.pagerank(nx_graph)

def cal_eigenvector(graph):
    graph.construct_weighted_clique()
    nx_graph = nx.convert_matrix.from_scipy_sparse_matrix(graph.weighted_matrix)

    return nx.eigenvector_centrality(nx_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=False, default="degree")
    parser.add_argument('--inputdir', required=True, type=str)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--with_self_loop', action='store_true')
    args = parser.parse_args()
    
    print(f'algo: {args.algo}, dataset: {args.dataset_name}', flush=True)

    graph = HyperGraph(args.inputdir, args.dataset_name, args.exist_hedgename, args.with_self_loop)
    if args.algo == "degree":
        node_centrality = cal_degree(graph)
    elif args.algo == "kcore":
        node_centrality = cal_kcore(graph)
    elif args.algo == "pagerank":
        node_centrality = cal_pagerank(graph)
    elif args.algo == "eigenvec":
        node_centrality = cal_eigenvector(graph)

    outputdir = os.path.join(args.inputdir, args.dataset_name)
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    outputname = os.path.join(outputdir, 
                              "{}_nodecentrality_self_loop.txt".format(args.algo) if args.with_self_loop \
                                  else "{}_nodecentrality.txt".format(args.algo)
                              )
    with open(outputname, "w") as f:
        f.write("node\t" + args.algo + "\n")
        for v in node_centrality:
            node_orgindex = graph.node_orgindex[v]
            f.write(str(node_orgindex) + "\t" + str(node_centrality[v]) + "\n")
            