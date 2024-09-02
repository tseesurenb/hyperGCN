'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="tempLGCN", description="Dynamic GCN-based CF recommender")
    parser.add_argument('--model', type=str, default='LightGCN', help='rec-model, support [LightGCN, NGCF, LightGCNAttn]')
    parser.add_argument('--dataset', type=str, default='ml-100k', help="available datasets: [ml100k, ml1m, ml10m]")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of GCN")
    parser.add_argument('--batch_size', type=int, default= 1024, help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int,default=21)
    parser.add_argument('--epochs_per_eval', type=int,default=10)
    parser.add_argument('--epochs_per_lr_decay', type=int,default=5000)
    parser.add_argument('--verbose', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-04, help="the weight decay for l2 normalizaton")
    parser.add_argument('--path', type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument('--top_k', type=int, default=20, help="@k test list")
    parser.add_argument('--r_beta', type=float, default=25)
    parser.add_argument('--a_beta', type=float, default=55)
    parser.add_argument('--a_method', type=str, default='exp')
    parser.add_argument('--r_method', type=str, default='exp')
    parser.add_argument('--enable_temp_emb', type=bool, default=False)
    parser.add_argument('--loadedModel', type=bool, default=False)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--pears_thresh', type=float, default=0.27)
    parser.add_argument('--u_sim_thresh', type=float, default=0.2)
    parser.add_argument('--i_sim_thresh', type=float, default=0.15)
    parser.add_argument('--u_sim', type=str, default='cosine')
    parser.add_argument('--i_sim', type=str, default='jaccard')
    parser.add_argument('--edge', type=str, default='bi')
    parser.add_argument('--u_sim_top_k', type=int, default=20)
    parser.add_argument('--i_sim_top_k', type=int, default=20)
    parser.add_argument('--self_sim', type=bool, default=False)
    parser.add_argument('--weight_mode', type=str, default='exp')
    parser.add_argument('--shuffle', type=bool, default=False)

    
    return parser.parse_args()