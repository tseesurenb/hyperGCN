'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import random
import matplotlib
matplotlib.use('Agg')
import sys

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import utils as ut

from sklearn.model_selection import train_test_split
from model import RecSysGNN
from sklearn import preprocessing as pp
from world import config
import data_prep as dp
from data_prep import get_edge_index, create_uuii_adjmat
import time
import world

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

        
def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0,  pos_emb0, neg_emb0):
    # compute loss from initial embeddings, used for regulization
            
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) + 
        pos_emb0.norm().pow(2)  +
        neg_emb0.norm().pow(2)
    ) / float(len(users))
    
    # compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
    return bpr_loss, reg_loss

def compute_bpr_loss_base(users, users_emb, pos_emb, neg_emb, users_base_emb, pos_base_emb, neg_base_emb, user_emb0,  pos_emb0, neg_emb0):
    # compute loss from initial embeddings, used for regulization
            
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) + 
        pos_emb0.norm().pow(2)  +
        neg_emb0.norm().pow(2) +
        users_base_emb.norm().pow(2) +
        pos_base_emb.norm().pow(2) +
        neg_base_emb.norm().pow(2)
    ) / float(len(users))
    
    users_emb = users_emb + users_base_emb
    pos_emb = pos_emb + pos_base_emb
    neg_emb = neg_emb + neg_base_emb
    
    # compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
    return bpr_loss, reg_loss

def train_and_eval(epochs, model, optimizer, train_df, train_neg_adj_list, test_df, batch_size, n_users, n_items, train_edge_index, train_edge_attrs, decay, topK, device, exp_n, g_seed):
   
    losses = {
        'loss': [],
        'bpr_loss': [],
        'reg_loss': []
    }

    metrics = {
        'recall': [],
        'precision': [],
        'f1': [],
        'ncdg': []      
    }

    pbar = tqdm(range(epochs), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    
    for epoch in pbar:
    
        final_loss_list, bpr_loss_list, reg_loss_list  = [], [], []

        S = ut.neg_uniform_sample(train_df, train_neg_adj_list, n_users)

        users = torch.Tensor(S[:, 0]).long().to(device)
        pos_items = torch.Tensor(S[:, 1]).long().to(device)
        neg_items = torch.Tensor(S[:, 2]).long().to(device)
        
        if config['shuffle']: 
            users, pos_items, neg_items = ut.shuffle(users, pos_items, neg_items)
        
        n_batch = len(users) // batch_size + 1
                            
        model.train()
        for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(ut.minibatch(users,
                                             pos_items,
                                             neg_items,
                                             batch_size=batch_size)):
                                     
            optimizer.zero_grad()

            if config['base'] == False:
                users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(batch_users, 
                                                                                                batch_pos, 
                                                                                                batch_neg, 
                                                                                                train_edge_index, 
                                                                                                train_edge_attrs)
                
                bpr_loss, reg_loss = compute_bpr_loss(
                    batch_users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
                )
            else:
                users_emb, pos_emb, neg_emb, users_base_emb, pos_base_emb, neg_base_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(batch_users, 
                                                                                                batch_pos, 
                                                                                                batch_neg, 
                                                                                                train_edge_index, 
                                                                                                train_edge_attrs)
                
                bpr_loss, reg_loss = compute_bpr_loss_base(
                    batch_users, users_emb, pos_emb, neg_emb, users_base_emb, pos_base_emb, neg_base_emb, userEmb0,  posEmb0, negEmb0
                )
            
            reg_loss = decay * reg_loss
            final_loss = bpr_loss + reg_loss
            
            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())
            
            # Update the description of the outer progress bar with batch information
            pbar.set_description(f'Exp {exp_n:2} | seed {g_seed:2} | #edges {len(train_edge_index[0]):6} | epoch({epochs}) {epoch} | Batch({n_batch}) {batch_i:3}')
            
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                _, out = model(train_edge_index, train_edge_attrs)
                final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
                test_topK_recall,  test_topK_precision, test_ncdg = ut.get_metrics(
                    final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, topK, device
                )
            
            if test_topK_recall + test_topK_precision != 0:
                f1 = (2 * test_topK_recall * test_topK_precision) / (test_topK_recall + test_topK_precision)
            else:
                f1 = 0.0
                
            losses['loss'].append(round(np.mean(final_loss_list),4))
            losses['bpr_loss'].append(round(np.mean(bpr_loss_list),4))
            losses['reg_loss'].append(round(np.mean(reg_loss_list),4))
            
            metrics['recall'].append(round(test_topK_recall,4))
            metrics['precision'].append(round(test_topK_precision,4))
            metrics['f1'].append(round(f1,4))
            metrics['ncdg'].append(round(test_ncdg,4))
            
            pbar.set_postfix_str(f"prec@20: {br}{test_topK_precision:.4f}{rs} | recall@20: {br}{test_topK_recall:.4f}{rs} | ncdg@20: {br}{test_ncdg:.4f}{rs}")
            pbar.refresh()

    return (losses, metrics)
    
def run_experiment(df, g_seed=42, exp_n = 1, device='cpu', verbose = -1):

    train_test_ratio = config['test_ratio']
    train, test = train_test_split(df.values, test_size=train_test_ratio, random_state=g_seed)
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)
    
    # Step 1: Make sure that the user and item pairs in the test set are also in the training set
    all_users = train_df['user_id'].unique()
    all_items = train_df['item_id'].unique()

    test_df = test_df[
      (test_df['user_id'].isin(all_users)) & \
      (test_df['item_id'].isin(all_items))
    ]
    
    # Step 2: Encode user and item IDs
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train_df['user_id'] = le_user.fit_transform(train_df['user_id'].values)
    train_df['item_id'] = le_item.fit_transform(train_df['item_id'].values)
    
    test_df['user_id'] = le_user.transform(test_df['user_id'].values)
    test_df['item_id'] = le_item.transform(test_df['item_id'].values)
    
    N_USERS = train_df['user_id'].nunique()
    N_ITEMS = train_df['item_id'].nunique()
    N_INTERACTIONS = len(train_df)
    #N_INTERACTIONS = len(train_df) + len(test_df)
    
    print(f"dataset: {br}{config['dataset']} {rs}| seed: {g_seed} | exp: {exp_n} | users: {N_USERS} | items: {N_ITEMS} | interactions: {N_INTERACTIONS}")
    
    # update the ids with new encoded values
    all_users = train_df['user_id'].unique()
    all_items = train_df['item_id'].unique()
    
    train_neg_adj_list = ut.make_neg_adj_list(train_df, all_items)
    #test_neg_adj_list = ut.make_neg_adj_list(test_df, all_items)
    
    # Step 3: Create edge index for user-to-item and item-to-user interactions
    u_t = torch.LongTensor(train_df.user_id)
    i_t = torch.LongTensor(train_df.item_id) + N_USERS

    # Step 4: Create bi-partite edge index
    bi_train_edge_index = torch.stack((
      torch.cat([u_t, i_t]),
      torch.cat([i_t, u_t])
    )).to(device)
    
    # Step 5: Create KNN user-to-user and item-to-item edge index     
    #knn_train_adj_df = create_uuii_adjmat_by_threshold(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_thresh=config['u_sim_thresh'], i_sim_thresh=config['i_sim_thresh'], self_sim=config['self_sim'])
    knn_train_adj_df = create_uuii_adjmat(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_top_k=config['u_sim_top_k'], i_sim_top_k=config['i_sim_top_k'], self_sim=config['self_sim']) 
    knn_train_edge_index, train_edge_attrs = get_edge_index(knn_train_adj_df)

    # Convert train_edge_index to a torch tensor if it's a numpy array
    if isinstance(knn_train_edge_index, np.ndarray):
        knn_train_edge_index = torch.tensor(knn_train_edge_index).to(device)
        knn_train_edge_index = knn_train_edge_index.long()
    
    # Concatenate user-to-user, item-to-item (from train_edge_index) and user-to-item, item-to-user (from train_edge_index2)
    if config['edge'] == 'full':
        train_edge_index = torch.cat((knn_train_edge_index, bi_train_edge_index), dim=1)
    elif config['edge'] == 'knn':
        train_edge_index = knn_train_edge_index
    elif config['edge'] == 'bi':
        train_edge_index = bi_train_edge_index # default to LightGCN
    
    train_edge_index = train_edge_index.clone().detach().to(device)
    train_edge_attrs = torch.tensor(train_edge_attrs).to(device)
    
    if verbose >= 1:
        print(f"bi edge len: {len(bi_train_edge_index[0])} | knn edge len: {len(knn_train_edge_index[0])} | full edge len: {len(train_edge_index[0])}")
        
    LATENT_DIM = config['emb_dim']
    N_LAYERS = config['layers']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    DECAY = config['decay']
    LR = config['lr']
    K = config['top_k']
    IS_TEMP = config['enable_temp_emb']
    MODEL = config['model']

    gcn_model = RecSysGNN(
      latent_dim=LATENT_DIM, 
      num_layers=N_LAYERS,
      num_users=N_USERS,
      num_items=N_ITEMS,
      model=MODEL,
      is_temp=IS_TEMP,
      weight_mode = config['weight_mode'],
      base = config['base']
    )
    gcn_model.to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=LR)

    losses, metrics = train_and_eval(EPOCHS, 
                                    gcn_model, 
                                    optimizer, 
                                    train_df,
                                    train_neg_adj_list,
                                    test_df,
                                    BATCH_SIZE, 
                                    N_USERS, 
                                    N_ITEMS, 
                                    train_edge_index, 
                                    train_edge_attrs, 
                                    DECAY, 
                                    K, 
                                    device, 
                                    exp_n, 
                                    g_seed)

   
    return losses, metrics


def run_experiment_2(train_df, test_df, g_seed=42, exp_n = 1, device='cpu', verbose = -1):

    
    train_df = filter_by_interactions(train_df, 10)
    
    all_users = train_df['user_id'].unique()
    all_items = train_df['item_id'].unique()
    
    test_df = test_df[
      (test_df['user_id'].isin(all_users)) & \
      (test_df['item_id'].isin(all_items))
    ]
    
    
    N_USERS = train_df['user_id'].nunique()
    N_ITEMS = train_df['item_id'].nunique()
    N_INTERACTIONS = len(train_df)
    
    print(f"dataset: {br}{config['dataset']} {rs}| seed: {g_seed} | exp: {exp_n} | users: {N_USERS} | items: {N_ITEMS} | interactions: {N_INTERACTIONS}")
    
    get_user_item_stats(train_df, test_df)
     
    train_adj_list = ut.make_neg_adj_list(train_df, all_items)
    #test_adj_list = ut.make_neg_adj_list(test_df, all_items)

    print("\nDone making adj list")
    
    u_t = torch.LongTensor(train_df.user_id)
    i_t = torch.LongTensor(train_df.item_id) + N_USERS

    if verbose >= 1:
        # Verify the ranges
        print("max user index: ", u_t.max().item(), "| min user index: ", u_t.min().item())
        print("max item index: ", i_t.max().item(), "| min item index:", i_t.min().item())

    bi_train_edge_index = torch.stack((
      torch.cat([u_t, i_t]),
      torch.cat([i_t, u_t])
    )).to(device)
    
    print("\nDone creating bi edge index")
         
    #knn_train_adj_df = create_uuii_adjmat_by_threshold(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_thresh=config['u_sim_thresh'], i_sim_thresh=config['i_sim_thresh'], self_sim=config['self_sim'])
    knn_train_adj_df = create_uuii_adjmat(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_top_k=config['u_sim_top_k'], i_sim_top_k=config['i_sim_top_k'], self_sim=config['self_sim'], verbose=verbose) 
    knn_train_edge_index, train_edge_attrs = get_edge_index(knn_train_adj_df)
    
    # Convert train_edge_index to a torch tensor if it's a numpy array
    if isinstance(knn_train_edge_index, np.ndarray):
        knn_train_edge_index = torch.tensor(knn_train_edge_index).to(device)
        knn_train_edge_index = knn_train_edge_index.long()
    
    # Concatenate user-to-user, item-to-item (from train_edge_index) and user-to-item, item-to-user (from train_edge_index2)
    if config['edge'] == 'full':
        train_edge_index = torch.cat((knn_train_edge_index, bi_train_edge_index), dim=1)
    elif config['edge'] == 'knn':
        train_edge_index = knn_train_edge_index
    elif config['edge'] == 'bi':
        train_edge_index = bi_train_edge_index # default to LightGCN
        print(f"Using bi edges and {len(bi_train_edge_index[0])} edges")
    
    train_edge_index = train_edge_index.clone().detach().to(device)
    train_edge_attrs = torch.tensor(train_edge_attrs).to(device)
    
    if verbose >= 1:
        print(f"bi edge len: {len(bi_train_edge_index[0])} | knn edge len: {len(knn_train_edge_index[0])} | full edge len: {len(train_edge_index[0])}")
    
    LATENT_DIM = config['emb_dim']
    N_LAYERS = config['layers']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    DECAY = config['decay']
    LR = config['lr']
    K = config['top_k']
    IS_TEMP = config['enable_temp_emb']
    MODEL = config['model']

    lightgcn = RecSysGNN(
      latent_dim=LATENT_DIM, 
      num_layers=N_LAYERS,
      num_users=N_USERS,
      num_items=N_ITEMS,
      model=MODEL,
      is_temp=IS_TEMP,
      weight_mode = config['weight_mode']
    )
    lightgcn.to(device)

    optimizer = torch.optim.Adam(lightgcn.parameters(), lr=LR)
    if verbose >=1:
        print("Size of Learnable Embedding : ", [x.shape for x in list(lightgcn.parameters())])

    #train_and_eval(epochs, model, optimizer, train_df, train_neg_adj_list, test_df, batch_size, n_users, n_items, train_edge_index, train_edge_attrs, decay, topK, device, exp_n, g_seed):
    losses, metrics = train_and_eval(EPOCHS, 
                                     lightgcn, 
                                     optimizer, 
                                     train_df,
                                     train_adj_list,
                                     test_df,
                                     BATCH_SIZE, 
                                     N_USERS, 
                                     N_ITEMS, 
                                     train_edge_index, 
                                     train_edge_attrs, 
                                     DECAY, 
                                     K, 
                                     device, 
                                     exp_n, 
                                     g_seed)

    #train_and_eval(epochs, model, optimizer, train_df, train_adj_list, test_df, test_adj_list, batch_size, n_users, n_items, train_edge_index, train_edge_attrs, decay, topK, device, exp_n, g_seed):
   
    return losses, metrics

def filter_by_interactions(df_selected, min_interactions):
    # Filter users with at least a minimum number of interactions
    user_interaction_counts = df_selected['user_id'].value_counts()
    filtered_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
    df_user_filtered = df_selected[df_selected['user_id'].isin(filtered_users)]

    # Filter items with at least a minimum number of interactions
    item_interaction_counts = df_user_filtered['item_id'].value_counts()
    filtered_items = item_interaction_counts[item_interaction_counts >= min_interactions].index
    df_filtered = df_user_filtered[df_user_filtered['item_id'].isin(filtered_items)]

    # Now, ensure users still have enough interactions after filtering items
    user_interaction_counts_after_item_filter = df_filtered['user_id'].value_counts()
    filtered_users_after_item_filter = user_interaction_counts_after_item_filter[user_interaction_counts_after_item_filter >= min_interactions].index
    df_filtered = df_filtered[df_filtered['user_id'].isin(filtered_users_after_item_filter)]

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    
    del df_selected
    del df_user_filtered
    
    return df_filtered.copy()
    

def get_user_item_stats(train_df, test_df):
        # Get user interaction statistics
        train_user_interactions = train_df.groupby('user_id').size()
        train_min_user_interactions = train_user_interactions.min()
        train_max_user_interactions = train_user_interactions.max()
        train_mean_user_interactions = round(train_user_interactions.mean(), 1)

        # Get item interaction statistics
        train_item_interactions = train_df.groupby('item_id').size()
        train_min_item_interactions = train_item_interactions.min()
        train_max_item_interactions = train_item_interactions.max()
        train_mean_item_interactions = round(train_item_interactions.mean(),1)
        
        # Get user interaction statistics
        test_user_interactions = test_df.groupby('user_id').size()
        test_min_user_interactions = test_user_interactions.min()
        test_max_user_interactions = test_user_interactions.max()
        test_mean_user_interactions = round(test_user_interactions.mean(), 1)

        # Get item interaction statistics
        test_user_interactions = train_df.groupby('item_id').size()
        test_min_item_interactions = test_user_interactions.min()
        test_max_item_interactions = test_user_interactions.max()
        test_mean_item_interactions = round(test_user_interactions.mean(),1)
        
        # update the ids with new encoded values
        all_users = train_df['user_id'].unique()
        all_items = train_df['item_id'].unique()
            

        print(f'trainset | min_user_interactions: {br}{train_min_user_interactions}{rs} | max_user_interactions: {br}{train_max_user_interactions}{rs} | mean_user_interactions: {br}{train_mean_user_interactions}{rs}')
        print(f'trainset | min_item_interactions: {br}{train_min_item_interactions}{rs} | max_item_interactions: {br}{train_max_item_interactions}{rs} | mean_item_interactions: {br}{train_mean_item_interactions}{rs}')   
        print(f' testset | min_user_interactions: {br}{test_min_user_interactions}{rs} | max_user_interactions: {br}{test_max_user_interactions}{rs} | mean_user_interactions: {br}{test_mean_user_interactions}{rs}')
        print(f' testset | min_item_interactions: {br}{test_min_item_interactions}{rs} | max_item_interactions: {br}{test_max_item_interactions}{rs} | mean_item_interactions: {br}{test_mean_item_interactions}{rs}')
                