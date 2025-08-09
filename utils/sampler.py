import dgl
import torch
from dgl.dataloading import BlockSampler
from dgl.transforms import to_block

# modified based on dgl.dataloading.NeighborSampler
class HypergraphNeighborSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # WARNING: in g, corresponding 'in' and 'con' edges should have the same ids! 
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            
            # including all already sampled edges and their reverse
            if len(blocks) > 0: 
                # add bidirectional edges
                tmp_eids = torch.unique(torch.concat([update_eid[etype] for etype in update_eid], dim=0))
                for etype in update_eid: 
                    update_eid[etype] = tmp_eids
                # combine previous eid and new sampled eid
                for etype in eid: 
                    update_eid[etype] = torch.unique(torch.concat((eid[etype], update_eid[etype]), dim=0))
                update_frontier = g.edge_subgraph(update_eid, relabel_nodes=False)
            else: 
                # add bidirectional edges
                tmp_eids = torch.concat([eid[etype] for etype in eid], dim=0)
                update_eid = {etype: tmp_eids for etype in eid}
                update_frontier = g.edge_subgraph(update_eid, relabel_nodes=False)
                block = to_block(update_frontier, seed_nodes)
                seed_nodes = block.srcdata[dgl.NID]
            
            block = to_block(update_frontier, seed_nodes)
            block.edata[dgl.EID] = update_eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
    
class HypergraphNeighborSamplerDeep(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # WARNING: in g, corresponding 'in' and 'con' edges should have the same ids! 
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[dgl.EID]
            
            # including all already sampled edges and their reverse
            if len(blocks) > 0: 
                # add bidirectional edges
                block = to_block(frontier, seed_nodes)
                seed_nodes = block.srcdata[dgl.NID]
                tmp_eids = torch.unique(torch.concat([update_eid[etype] for etype in update_eid] + [eid[etype] for etype in eid], dim=0))
                for etype in update_eid: 
                    update_eid[etype] = tmp_eids
                update_frontier = g.edge_subgraph(update_eid, relabel_nodes=False)
            else: 
                # add bidirectional edges
                tmp_eids = torch.concat([eid[etype] for etype in eid], dim=0)
                update_eid = {etype: tmp_eids for etype in eid}
                update_frontier = g.edge_subgraph(update_eid, relabel_nodes=False)
                block = to_block(update_frontier, seed_nodes)
                seed_nodes = block.srcdata[dgl.NID]
            
            block = to_block(update_frontier, seed_nodes)
            block.edata[dgl.EID] = update_eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
