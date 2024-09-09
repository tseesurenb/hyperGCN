'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import plot_results, print_metrics, set_seed
from procedure import run_experiment
import data_prep as dp 
from world import config
import pickle
import os

# STEP 1: set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# STEP 2: Load the data and filter only ratings >= 3
if config['dataset'] == 'ml-100k' or config['dataset'] == 'ml-1m':
    min_interactions = 0
else:
    min_interactions = 20
    
df, u_df, i_df, stats = dp.load_data(dataset = config['dataset'], u_min_interaction_threshold = min_interactions, i_min_interaction_threshold = min_interactions, verbose=config['verbose'])
df = df[df['rating']>=3] # How many ratings are a 3 or above?
        
#seeds = [2020, 12, 89, 91, 41]
seeds = [2020]

old_edge_type = config['edge']
old_model_type = config['model']

config['edge'] = 'bi'
config['model'] = 'LightGCN'

all_bi_metrics, all_bi_losses = [], []
recalls, precs, f1s, ncdg,  = [], [], [], []
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
 
        set_seed(seed)
 
        losses, metrics = run_experiment(df = df, g_seed = seed, exp_n = exp_n, device=device, verbose=config['verbose'])
        
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
    
num_hyphens = 100  # Adjust the number of hyphens as needed
print(f"\n{'-' * num_hyphens}\n")  

config['edge'] = old_edge_type
config['model'] = old_model_type

file_name = f"models/{config['model']}_{config['dataset']}_{config['edge']}_{config['weight_mode']}_{config['layers']}_{config['epochs']}"

all_knn_metrics, all_knn_losses = [], []
recalls, precs, f1s, ncdg = [], [], [], []

for seed in seeds:
    
    set_seed(seed)
    
    losses, metrics = run_experiment(df = df, g_seed = seed, exp_n = exp_n, device=device, verbose=-1)
    
    max_idx = np.argmax(metrics['f1'])
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    ncdg.append(np.max(metrics['ncdg']))
    all_knn_losses.append(losses)
    all_knn_metrics.append(metrics)
    
    exp_n += 1
   
print_metrics(recalls, precs, f1s, ncdg, stats=stats)
plot_results(file_name, len(seeds), config['epochs'], all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics)