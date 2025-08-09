import os
import random
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', required=True, type=str)
    parser.add_argument('--outputdir', required=True, type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--output_dim', required=True, type=int)
    parser.add_argument('--seed', required=False, default=0)
    # parser.add_argument('--gen_edge_feat', action='store_true')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if not os.path.exists("%s/dist_gaussian" % args.outputdir):
        os.makedirs("%s/dist_gaussian" % args.outputdir)

    savefname = "%s/dist_gaussian/%s_dim_%d.npy" % (
        args.outputdir, args.dataset_name, args.output_dim)
    saveOriginIndexfname = "%s/dist_gaussian/%s_dim_%d_origin_index.txt" % (
        args.outputdir, args.dataset_name, args.output_dim)
    
    # if not args.gen_edge_feat: 
    #     savefname = "%s/dist_gaussian/%s_dim_%d.npy" % (
    #         args.outputdir, args.dataset_name, args.output_dim)
    #     saveOriginIndexfname = "%s/dist_gaussian/%s_dim_%d_origin_index.txt" % (
    #         args.outputdir, args.dataset_name, args.output_dim)
    # else: 
    #     savefname = "%s/dist_gaussian/%s_dim_%d_edge.npy" % (
    #         args.outputdir, args.dataset_name, args.output_dim)
    #     saveOriginIndexfname = "%s/dist_gaussian/%s_dim_%d_edge_origin_index.txt" % (
    #         args.outputdir, args.dataset_name, args.output_dim)
    
    nodename2id = dict()
    node_org_index = []
    node_label_count = []
    with open(os.path.join(args.inputdir, args.dataset_name, 'hypergraph.txt'), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    if args.exist_hedgename:
        lines = [line[1:] for line in lines]
    hedge2node = [[int(node) for node in line] for line in lines]

    with open(os.path.join(args.inputdir, args.dataset_name, 'hypergraph_pos.txt'), 'r') as f: 
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    if args.exist_hedgename:
        lines = [line[1:] for line in lines]
    hedge2nodepos = [[int(pos) for pos in line] for line in lines]

    for i, nodes in enumerate(hedge2node):
        for j, node in enumerate(nodes):
            if node not in nodename2id:
                nodename2id[node] = len(nodename2id)
                node_org_index.append(node)
                node_label_count.append([0]*args.output_dim)
            node_label_count[nodename2id[node]][hedge2nodepos[i][j]] += 1

    node_label_dist = np.array(node_label_count)
    node_label_dist = node_label_dist / node_label_dist.sum(axis=1, keepdims=True)
    node_label_dist_gaussian = node_label_dist + np.random.normal(0, 1/args.output_dim, size=node_label_dist.shape)

    np.save(savefname, node_label_dist)
    print("save feature file success! ", flush=True)
    print(savefname, flush=True)

    with open(saveOriginIndexfname, 'w') as f: 
        f.writelines([str(index)+'\n' for index in node_org_index])

    print('done.', flush=True)
