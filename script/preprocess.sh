#!/bin/bash
cd ..
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "AMinerAuthor" --feat_dim 44
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "DBLP" --feat_dim 44 --exist_hedgename
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "emailEnron" --feat_dim 44
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "emailEu" --feat_dim 44
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "StackOverflowBiology" --feat_dim 44
python preprocess/initialize_by_rw.py --inputdir "dataset/edge_dependent_node_classification" --outputdir "features/edge_dependent_node_classification" --dataset_name "StacOverflowPhysics" --feat_dim 44

# preprocessing is only needed for datasets without original node features
# for the outsider identification datasets, the initial node features have been included in the `features` folder, which are the same features as in https://github.com/Graph-COM/ED-HNN

