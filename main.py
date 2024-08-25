import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import run_experiment, plot_loss2, plot_loss3
import data_prep as dp 
from world import config
import pickle
import os

# STEP 1: set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# STEP 2: Load the data and filter only ratings >= 3
columns_name=['user_id','item_id','rating','timestamp']
df = pd.read_csv("data/raw/ml-100k/u.data",sep="\t",names=columns_name)
df = df[df['rating']>=3] # How many ratings are a 3 or above?

num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
num_interactions = len(df)
seeds = [7, 12, 89, 91, 41]
#seeds = [7]

recalls = []
precs = []
f1s = []
ncdg = []
exp_n = 1

edge_value = config['edge']
config['edge'] = 'bi'

all_bi_metrics = []
all_bi_losses = []

file_path = f"models/{config['model']}_{config['edge']}_{config['layers']}_{config['epochs']}_experiment_results.pkl"

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

        losses, metrics = run_experiment(df = df, g_seed = seed, exp_n = exp_n, device=device, verbose=-1)
        
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
    with open(f"models/{config['model']}_{config['edge']}_{config['layers']}_{config['epochs']}_experiment_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)

print(f'\n-------------------------------------------- Experimental Results --------------------------------------------')    
print(f"Dataset: {config['dataset']}, num_users: {num_users}, num_items: {num_items}, num_interactions: {num_interactions}")
print(f"MODEL: {config['model']} | EDGE TYPE: {config['edge']} | LATENT_DIM: {config['emb_dim']} | N_LAYERS: {config['layers']} | SIMILARITY: u - {config['u_sim']} with {config['u_sim_thresh']}, i - {config['i_sim']} with {config['i_sim_thresh']} | BATCH_SIZE: {config['batch_size']} | DECAY: {config['decay']} | topK: {config['top_k']} | IS_TEMP: {config['enable_temp_emb']}")
print(f"  Recall: {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} | {round(np.mean(recalls), 4):.4f}, {round(np.std(recalls), 4):.4f}")
print(f"    Prec: {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} | {round(np.mean(precs), 4):.4f}, {round(np.std(precs), 4):.4f}")
print(f"F1 score: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}, {f1s[3]:.4f}, {f1s[4]:.4f} | {round(np.mean(f1s), 4):.4f}, {round(np.std(f1s), 4):.4f}")
print(f"NDCG: {ncdg[0]:.4f}, {ncdg[1]:.4f}, {ncdg[2]:.4f}, {ncdg[3]:.4f}, {ncdg[4]:.4f} | {round(np.mean(ncdg), 4):.4f}, {round(np.std(ncdg), 4):.4f}")

config['edge'] = edge_value

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

    losses, metrics = run_experiment(df = df, g_seed = seed, exp_n = exp_n, device=device, verbose=-1)
    
    max_idx = np.argmax(metrics['f1'])
    #all_metrics.append(metrics)
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    ncdg.append(np.max(metrics['ncdg']))
    all_knn_losses.append(losses)
    all_knn_metrics.append(metrics)
    
    exp_n += 1

print(f'\n-------------------------------------------- Experimental Results --------------------------------------------')    
print(f"Dataset: {config['dataset']}, num_users: {num_users}, num_items: {num_items}, num_interactions: {num_interactions}")
print(f"MODEL: {config['model']} | EDGE TYPE: {config['edge']} | LATENT_DIM: {config['emb_dim']} | N_LAYERS: {config['layers']} | SIMILARITY: u - {config['u_sim']} with {config['u_sim_thresh']}, i - {config['i_sim']} with {config['i_sim_thresh']} | BATCH_SIZE: {config['batch_size']} | DECAY: {config['decay']} | topK: {config['top_k']} | IS_TEMP: {config['enable_temp_emb']}")
print(f"  Recall: {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} | {round(np.mean(recalls), 4):.4f}, {round(np.std(recalls), 4):.4f}")
print(f"    Prec: {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} | {round(np.mean(precs), 4):.4f}, {round(np.std(precs), 4):.4f}")
print(f"F1 score: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}, {f1s[3]:.4f}, {f1s[4]:.4f} | {round(np.mean(f1s), 4):.4f}, {round(np.std(f1s), 4):.4f}")
print(f"NDCG: {ncdg[0]:.4f}, {ncdg[1]:.4f}, {ncdg[2]:.4f}, {ncdg[3]:.4f}, {ncdg[4]:.4f} | {round(np.mean(ncdg), 4):.4f}, {round(np.std(ncdg), 4):.4f}")

plot_loss3(len(seeds), config['epochs'], all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics)