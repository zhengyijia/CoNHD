import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--task_type', default=None, type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_runs', required=False, default=10, type=int, help='number of runs for node classification task')
    parser.add_argument('--specify_num_run', required=False, default=None, type=int, help='excute a specific run in num_runs')
    parser.add_argument('--recalculate', action='store_true')

    # training parameter
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--input_dropout', default=0, type=float)
    parser.add_argument('--input_vfeat_dropout', default=0, type=float)
    parser.add_argument('--valid_epoch', default=5, type=int)
    parser.add_argument('--save_best_epoch', action='store_true')

    # data parameter
    parser.add_argument('--data_inputdir', default='dataset/', type=str)
    parser.add_argument('--add_self_loop', default='None', choices=['None', 'All', 'Isolated'])
    parser.add_argument('--feature_inputdir', default='features/', type=str)
    parser.add_argument('--dataset_name', default='DBLP', type=str)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--evaltype', default='valid', type=str)
    parser.add_argument('--node_sampling', default=-1, type=int) # node -> hedge
    parser.add_argument('--hedge_sampling', default=-1, type=int) # hedge -> node
    parser.add_argument('--valid_inputname', default='valid_hindex', type=str)
    parser.add_argument('--test_inputname', default='test_hindex', type=str)
    parser.add_argument('--init_feat_type', default='rw', type=str)
    parser.add_argument('--init_feat_dim', default=0, type=int)
    parser.add_argument('--init_feat_sparse', action='store_true')
    parser.add_argument('--feat_scale_transform', action='store_true')
    parser.add_argument('--fix_init_embedder', action='store_true')

    # parameter for train_with_edge_feat
    parser.add_argument('--init_feat_type_edge', default='rw', type=str)
    parser.add_argument('--init_feat_dim_edge', default=0, type=int)
    parser.add_argument('--init_feat_sparse_edge', action='store_true')
    parser.add_argument('--feat_scale_transform_edge', action='store_true')
    
    # weight for HNHN
    parser.add_argument('--use_exp_wt', action='store_true')
    parser.add_argument('--alpha_e', default=0, type=float)
    parser.add_argument('--alpha_v', default=0, type=float)
    
    # HNN
    parser.add_argument('--psi_num_layers', default=1, type=int)
    parser.add_argument('--efeat', default='zero', type=str, help="initialize for hyperedge embedding")
    
    # model parameter
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--PE_Block', default="ISAB", choices=("ISAB", "SAB", "UNP"), type=str, help="Permutation Equavariant Block")
    parser.add_argument('--num_inds', default=4, type=int)
    parser.add_argument('--embedder', default='hnhn', type=str)
    parser.add_argument('--scorer', default='sm', type=str)
    parser.add_argument('--scorer_num_layers', default=1, type=int)
    parser.add_argument('--scorer_hidden_dim', default=256, type=int)
    parser.add_argument('--att_type_v', default='', type=str, help="OrderPE, ITRE, ShawRE, pure, NoAtt")
    parser.add_argument('--agg_type_v', default='', type=str, help="PrevQ, pure, pure2")
    parser.add_argument('--att_type_e', default='', type=str, help="OrderPE, pure, NoAtt")
    parser.add_argument('--agg_type_e', default='', type=str, help="PrevQ, pure, pure2")
    parser.add_argument('--dim_vertex', default=128, type=int)
    parser.add_argument('--dim_edge', default=128, type=int)
    parser.add_argument('--dim_hidden', default=256, type=int)
    # pe
    parser.add_argument('--vorder_input', default='', type=str, help="positional encoding input for OrderPE")

    # CoNHD
    parser.add_argument('--num_att_layer', default=1, type=int, help="Set the number of Self-Attention layers")
    parser.add_argument('--co_rep_dim', default=128, type=int, 
                        help='Dimension of co-representation in TwoLevelAttNet')
    parser.add_argument('--layer_dim_hidden', default=128, type=int)
    parser.add_argument('--layernorm', action='store_true')
    parser.add_argument('--not_share_weights', action='store_true')

    # CoNHD_Ablation
    parser.add_argument('--node_agg', action='store_true')  # node -> hedge
    parser.add_argument('--hedge_agg', action='store_true')  # hedge -> node

    # TCoNHDNodeClassification
    parser.add_argument('--output_agg', default='feat_mean', choices=('feat_mean', 'pred_mean', 'feat_att'))

    # EDHNN
    parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')

    # run_diffusion
    parser.add_argument('--diff_lr', type=float, default=0.1)
    parser.add_argument('--diff_bs', type=int, default=-1)
    parser.add_argument('--diff_edge_reg_type', type=str, default='ce', choices=('ce', 'tv', 'lec'))
    parser.add_argument('--diff_node_reg_type', type=str, default='ce', choices=('ce', 'tv', 'lec'))
    parser.add_argument('--diff_lambda', type=float, default=0.1)
    parser.add_argument('--diff_gamma', type=float, default=0.1)

    # fit diffusion
    parser.add_argument('--optim', choices=['GD', 'ADMM'], default='GD', required=False)
    parser.add_argument('--hedge_reg', choices=['ce', 'tv', 'lec', 'origin'], default='ce', required=False)
    parser.add_argument('--node_reg', choices=['ce', 'tv', 'lec'], default='ce', required=False)
    parser.add_argument('--feat_dim', required=False, default=1, type=int)
    parser.add_argument('--nstep', required=False, default=2, type=int)

    args = parser.parse_args()
    
    # vorder
    if len(args.vorder_input) == 0:
        args.vorder_input = []
        args.orderflag = False
    else:
        args.vorder_input = args.vorder_input.split(",")
        args.orderflag = True
    args.order_dim = len(args.vorder_input)
    
    args.share_weights = not args.not_share_weights
    
    # Setting File Save Name -----------------------------------------------------------------------------
    args.embedder_name = args.embedder
            
    if len(args.att_type_v) > 0 and len(args.agg_type_v) > 0:
        args.embedder_name += "-{}-{}".format(args.att_type_v, args.agg_type_v)
    if len(args.att_type_e) > 0 and len(args.agg_type_e) > 0:
        args.embedder_name += "-{}-{}".format(args.att_type_e, args.agg_type_e)
    if len(args.att_type_v) > 0 and args.att_type_v != "NoAtt":
        args.embedder_name += "_atnl{}".format(args.num_att_layer)
    elif len(args.att_type_e) > 0 and args.att_type_e != "NoAtt":
        args.embedder_name += "_atnl{}".format(args.num_att_layer)
    args.embedder_name += "_nl{}".format(args.num_layers)
    
    args.scorer_name = "{}_snl{}".format(args.scorer, args.scorer_num_layers)
    args.model_name = args.embedder_name + "_" + args.scorer_name
    if args.embedder == "CoNHD_GD" or args.embedder == "CoNHD_ADMM": 
        args.model_name += "_" + args.PE_Block
    
    if args.embedder == "hcha":
        args.param_name = "hd_{}_od_{}_do_{}_lr_{}_ni_{}_nsp_{}_hsp_{}".format(args.dim_hidden, args.dim_edge, args.dropout, args.lr, args.num_inds, args.node_sampling, args.hedge_sampling)
    elif args.embedder == "hnn":
        args.param_name = "hd_{}_od_{}_do_{}_lr_{}_psi_{}_ie_{}_nsp_{}_hsp_{}".format(args.dim_hidden, args.dim_edge, args.dropout, args.lr, args.psi_num_layers, args.efeat, args.node_sampling, args.hedge_sampling)
    elif args.embedder == "CoNHD_GD" or args.embedder == "CoNHD_ADMM":
        args.param_name = "hd_{}_od_{}_do_{}_bs_{}_lr_{}_ni_{}_nsp_{}_hsp_{}".format(args.dim_hidden, args.co_rep_dim, args.dropout, args.bs, args.lr, args.num_inds, args.node_sampling, args.hedge_sampling)
    else:
        args.param_name = "hd_{}_od_{}_do_{}_bs_{}_lr_{}_ni_{}_nsp_{}_hsp_{}".format(args.dim_hidden, args.dim_edge, args.dropout, args.bs, args.lr, args.num_inds, args.node_sampling, args.hedge_sampling)
    
    if args.layernorm: 
        args.param_name += '_ln'

    if args.input_dropout: 
        args.param_name += f"_ido_{args.input_dropout}"

    if args.input_vfeat_dropout:
        args.param_name += f"_ivdo_{args.input_vfeat_dropout}"

    if args.node_agg: 
        args.param_name += "_node_agg"
    if args.hedge_agg: 
        args.param_name += "_hedge_agg"

    if args.not_share_weights:
        args.param_name += "_not_share_weights"

    print(args, flush=True)

    # ---------------------------------------------------------------------------------------------------
    return args
