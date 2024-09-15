'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from procedure import run_experiment_2
from utils import plot_results, print_metrics, set_seed
import data_prep as dp 
from world import config
import pickle
import os

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

# STEP 1: set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# STEP 2: Load the data and filter only ratings >= 3
train_df, test_df = dp.load_data_from_adj_list(dataset = config['dataset'])

num_users = train_df['user_id'].nunique()
num_items = train_df['item_id'].nunique()
num_interactions = len(train_df)

stats = {'num_users': num_users, 'num_items': num_items,  'num_interactions': num_interactions}

#seeds = [2020, 12, 89, 91, 41]
seeds = [2020]

#old_edge_type = config['edge']
#old_model_type = config['model']

#config['edge'] = 'bi'
#config['model'] = 'LightGCN'

all_bi_metrics = []
all_bi_losses = []

recalls = []
precs = []
f1s = []
ncdg = []
exp_n = 1

for seed in seeds:
    #print(f'Experiment ({exp_n}) starts with seed:{seed}')
    
    set_seed(seed)
    
    #edges = dp.get_edges(df)

    losses, metrics = run_experiment_2(o_train_df = train_df, o_test_df=test_df, g_seed = seed, exp_n = exp_n, device=device, verbose=config['verbose'])
    
    max_idx = np.argmax(metrics['f1'])
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    ncdg.append(np.max(metrics['ncdg']))
    all_bi_losses.append(losses)
    all_bi_metrics.append(metrics)
    
    exp_n += 1
    
# Assuming you have the following lists to save
all_results = {
    'recalls': recalls,
    'precs': precs,
    'f1s': f1s,
    'ncdg': ncdg,
    'all_bi_losses': all_bi_losses,
    'all_bi_metrics': all_bi_metrics
}


print_metrics(recalls, precs, f1s, ncdg, stats=stats)

