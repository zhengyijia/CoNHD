#!/bin/bash
cd ../..
seedlist=("0" "10" "100" "500" "10000")
for seed in ${seedlist[@]}
do
    # CoNHD_GD (UNB)
    python train.py --dataset_name citeseer_cocitation_outsider_id --init_feat_type origin --init_feat_dim 3703 --output_dim 2 --embedder CoNHD_GD --att_type_v pure --agg_type_v PrevQ --att_type_e pure --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer im --scorer_num_layers 1 --bs 64 --lr 0.001 --node_sampling -1 --hedge_sampling 40 --dropout 0.7 --input_dropout 0.7 --optimizer "adam" --gamma 0.99 --dim_hidden 64 --epochs 100 --valid_epoch 5 --evaltype test --save_best_epoch --seed ${seed} --co_rep_dim 128 --PE_Block UNP --use_gpu --fix_init_embedder
    # CoNHD_GD (ISAB)
    python train.py --dataset_name citeseer_cocitation_outsider_id --init_feat_type origin --init_feat_dim 3703 --output_dim 2 --embedder CoNHD_GD --att_type_v pure --agg_type_v PrevQ --att_type_e pure --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer im --scorer_num_layers 1 --bs 64 --lr 0.001 --node_sampling -1 --hedge_sampling 40 --dropout 0.7 --input_dropout 0.7 --optimizer "adam" --gamma 0.99 --dim_hidden 64 --epochs 100 --valid_epoch 5 --evaltype test --save_best_epoch --seed ${seed} --co_rep_dim 128 --PE_Block ISAB --use_gpu --fix_init_embedder
done
