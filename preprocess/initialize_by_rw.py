import os
import random
import argparse
import numpy as np
import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec

from data_load import HyperGraph
from random_walk_hyper import random_walk_hyper
from deep_walk_hyper import deep_walk_hyper

# code from https://github.com/ma-compbio/Hyper-SAGNN
def walkpath2str(walk):
	return [list(map(str, w)) for w in walk]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', required=True, type=str)
    parser.add_argument('--outputdir', required=True, type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--feat_dim', required=True, type=int)
    parser.add_argument('--with_self_loop', action='store_true')
    parser.add_argument('--p', type=float, default=2, help='Return hyperparameter')
    parser.add_argument('--q', type=float, default=0.25, help='Inout hyperparameter')
    parser.add_argument('-l', '--walk_length', type=int, default=40, help='Length of walk per source')
    parser.add_argument('-r', '--num_walks', type=int, default=10, help='Number of walks per source')
    parser.add_argument('-k', '--window_size', type=int, default=10, help='Context size for optimization')
    parser.add_argument('--seed', required=False, default=0)
    parser.add_argument('--gen_edge_feat', action='store_true')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists("%s/rw" % args.outputdir):
        os.makedirs("%s/rw" % args.outputdir)
    
    if not args.gen_edge_feat: 
        savefname = "%s/rw/%s_dim_%d.npy" % (
            args.outputdir, args.dataset_name, args.feat_dim)
        saveOriginIndexfname = "%s/rw/%s_dim_%d_origin_index.txt" % (
            args.outputdir, args.dataset_name, args.feat_dim)
    else: 
        savefname = "%s/rw/%s_dim_%d_edge.npy" % (
            args.outputdir, args.dataset_name, args.feat_dim)
        saveOriginIndexfname = "%s/rw/%s_dim_%d_edge_origin_index.txt" % (
            args.outputdir, args.dataset_name, args.feat_dim)
    
    data = HyperGraph(args.inputdir, args.dataset_name, args.exist_hedgename, args.with_self_loop)
    
    node_list = np.arange(data.numnodes).astype('int')
    hedge_list = np.arange(data.numhedges).astype('int')
    
    if os.path.isfile(savefname):
        print("feature file exist! ", flush=True)
        print(savefname, flush=True)
    else: 
        if not args.gen_edge_feat: 
            walk_path = random_walk_hyper(args, node_list, data.hedge2node)
            # walk_path = deep_walk_hyper(args, node_list, data.hedge2node)
        else: 
            # use deep walk instead of random walk for edge feature to reduce memory usage
            walk_path = random_walk_hyper(args, hedge_list, data.node2hedge, gen_edge_feat=True)
            # walk_path = deep_walk_hyper(args, hedge_list, data.node2hedge, gen_edge_feat=True)
        walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
        print("Start turning path to strs")
        split_num = 20
        pool = ProcessPoolExecutor(max_workers=split_num)
        process_list = []
        walks = np.array_split(walks, split_num)
        result = []
        for walk in walks:
            process_list.append(pool.submit(walkpath2str, walk))
        for p in as_completed(process_list):
            result += p.result()
        pool.shutdown(wait=True)
        walks = result
        # print(walks)
        print("Start Word2vec")
        print("num cpu cores", multiprocessing.cpu_count())
        w2v = Word2Vec(walks, vector_size=args.feat_dim, window=args.window_size, min_count=0, sg=1, epochs=1, workers=multiprocessing.cpu_count())
        print(w2v.wv['0'])
        wv = w2v.wv
        if not args.gen_edge_feat:
            A = [wv[str(i)] for i in range(data.numnodes)]
        else: 
            A = [wv[str(i)] for i in range(data.numhedges)]
        np.save(savefname, A)
        print("save feature file success! ", flush=True)
        print(savefname, flush=True)
        
    if os.path.isfile(saveOriginIndexfname): 
        print("origin index file exist! ", flush=True)
        print(saveOriginIndexfname, flush=True)
    else: 
        if not args.gen_edge_feat: 
            OriginIndexes = [data.node_orgindex[index]+'\n' for index in node_list]
        else:
            OriginIndexes = [str(data.hedgename[index])+'\n' for index in hedge_list]
        with open(saveOriginIndexfname, 'w') as f: 
            f.writelines(OriginIndexes)
        print("save origin index file success! ", flush=True)
        print(saveOriginIndexfname, flush=True)
    
    print('done.', flush=True)
