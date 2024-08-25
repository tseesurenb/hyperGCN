'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

import os
from os.path import join
from enum import Enum
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['layers'] = args.layers
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_k'] = args.top_k
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['epochs_per_lr_decay'] = args.epochs_per_lr_decay
config['seed'] = args.seed
config['r_beta'] = args.r_beta
config['a_beta'] = args.a_beta
config['a_method'] = args.a_method
config['r_method'] = args.r_method
config['enable_temp_emb'] = args.enable_temp_emb
config['loadedModel'] = args.loadedModel
config['test_ratio'] = args.test_ratio
config['pears_thresh'] = args.pears_thresh
config['u_sim_thresh'] = args.u_sim_thresh
config['i_sim_thresh'] = args.i_sim_thresh
config['u_sim'] = args.u_sim
config['i_sim'] = args.i_sim
config['edge'] = args.edge
