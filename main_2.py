'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import run_experiment, run_experiment_2, plot_results
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

seeds = [7, 12, 89, 91, 41]
#seeds = [7]

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

print(f" Dataset: {config['dataset']}, num_users: {num_users}, num_items: {num_items}, num_interactions: {num_interactions}")
print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | EMB_DIM: {br}{config['emb_dim']}{rs} | #LAYERS: {br}{config['layers']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs}")
print(f"  Recall: {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} | {round(np.mean(recalls), 4):.4f}, {round(np.std(recalls), 4):.4f}")
print(f"    Prec: {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} | {round(np.mean(precs), 4):.4f}, {round(np.std(precs), 4):.4f}")
print(f"F1 score: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}, {f1s[3]:.4f}, {f1s[4]:.4f} | {bb}{round(np.mean(f1s), 4):.4f}{rs}, {round(np.std(f1s), 4):.4f}")
print(f"    NDCG: {ncdg[0]:.4f}, {ncdg[1]:.4f}, {ncdg[2]:.4f}, {ncdg[3]:.4f}, {ncdg[4]:.4f} | {bb}{round(np.mean(ncdg), 4):.4f}{rs}, {round(np.std(ncdg), 4):.4f}")
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
   
print(f"Dataset: {config['dataset']}, num_users: {num_users}, num_items: {num_items}, num_interactions: {num_interactions}")
print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | EMB_DIM: {br}{config['emb_dim']}{rs} | #LAYERS: {br}{config['layers']}{rs} | SIM: {br}u-{config['u_sim']}(topK {config['u_sim_top_k']}), i-{config['i_sim']}(topK {config['i_sim_top_k']}){rs} | Weight mode: {br}{config['weight_mode']}{rs} | Self-sim: {br}{config['self_sim']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs}")
print(f"  Recall: {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} | {round(np.mean(recalls), 4):.4f}, {round(np.std(recalls), 4):.4f}")
print(f"    Prec: {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} | {round(np.mean(precs), 4):.4f}, {round(np.std(precs), 4):.4f}")
print(f"F1 score: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}, {f1s[3]:.4f}, {f1s[4]:.4f} | {bb}{round(np.mean(f1s), 4):.4f}{rs}, {round(np.std(f1s), 4):.4f}")
print(f"    NDCG: {ncdg[0]:.4f}, {ncdg[1]:.4f}, {ncdg[2]:.4f}, {ncdg[3]:.4f}, {ncdg[4]:.4f} | {bb}{round(np.mean(ncdg), 4):.4f}{rs}, {round(np.std(ncdg), 4):.4f}")

plot_loss3(file_name, len(seeds), config['epochs'], all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics)