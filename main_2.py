'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import run_experiment, plot_results, print_metrics
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

#seeds = [7, 12, 89, 91, 41]
seeds = [7]

old_edge_type = config['edge']
old_model_type = config['model']

config['edge'] = 'bi'
config['model'] = 'LightGCN'

all_bi_metrics = []
all_bi_losses = []

recalls = []
precs = []
f1s = []
ncdg = []
exp_n = 1

file_name = f"models/{config['model']}_{config['dataset']}_{config['edge']}_{config['layers']}_{config['epochs']}"
file_path = file_name + "_experiment_results.pkl"

# Check if the results file exists
if os.path.exists(file_path):
    print(f"Loading results from {file_path}...")
    
    # Load the results
    with open(file_path, 'rb') as f:
        all_results = pickle.load(f)

    # Unpack the loaded results
    recalls = all_results['recalls']
    precs = all_results['precs']
    f1s = all_results['f1s']
    ncdg = all_results['ncdg']
    all_bi_losses = all_results['all_bi_losses']
    all_bi_metrics = all_results['all_bi_metrics']
else:
    for seed in seeds:
        #print(f'Experiment ({exp_n}) starts with seed:{seed}')
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        #edges = dp.get_edges(df)

        losses, metrics = run_experiment_2(train_df = train_df, test_df=test_df, g_seed = seed, exp_n = exp_n, device=device, verbose=config['verbose'])
        
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

    # Save to a file
    with open(file_path, 'wb') as f:
        pickle.dump(all_results, f)

print_metrics(recalls, precs, f1s, ncdg, stats=stats)

print(f'\n----------------------------------------------------------------------------------------\n')    

config['edge'] = old_edge_type
config['model'] = old_model_type

file_name = f"models/{config['model']}_{config['dataset']}_{config['edge']}_{config['weight_mode']}_{config['layers']}_{config['epochs']}"

all_knn_metrics = []
all_knn_losses = []

recalls = []
precs = []
f1s = []
ncdg = []

for seed in seeds:
    #print(f'Experiment ({exp_n}) starts with seed:{seed}')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #edges = dp.get_edges(df)

    losses, metrics = run_experiment_2(train_df = train_df, test_df=test_df, g_seed = seed, exp_n = exp_n, device=device, verbose=config['verbose'])
    
    max_idx = np.argmax(metrics['f1'])
    #all_metrics.append(metrics)
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    ncdg.append(np.max(metrics['ncdg']))
    all_knn_losses.append(losses)
    all_knn_metrics.append(metrics)
    
    exp_n += 1
   
print_metrics(recalls, precs, f1s, ncdg, stats=stats)
plot_results(file_name, len(seeds), config['epochs'], all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics)