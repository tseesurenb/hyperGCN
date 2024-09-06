'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import random
import matplotlib
matplotlib.use('Agg')

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from model import RecSysGNN
from sklearn import preprocessing as pp
from world import config
import data_prep as dp
from data_prep import get_edge_index, create_uuii_adjmat
import time
import sys

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def print_metrics(recalls, precs, f1s, ncdg, stats): 
    
    print(f" Dataset: {config['dataset']}, num_users: {stats['num_users']}, num_items: {stats['num_items']}, num_interactions: {stats['num_interactions']}")
    
    if config['edge'] == 'bi':
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs}")
    else:
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | SIM (mode-{config['weight_mode']}, self-{config['self_sim']}): {br}u-{config['u_sim']}(topK {config['u_sim_top_k']}), i-{config['i_sim']}(topK {config['i_sim_top_k']}){rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs}")

    metrics = [("Recall", recalls), 
           ("Prec", precs), 
           ("F1 score", f1s), 
           ("NDCG", ncdg)]

    for name, metric in metrics:
        values_str = ', '.join([f"{x:.4f}" for x in metric[:5]])
        mean_str = f"{round(np.mean(metric), 4):.4f}"
        std_str = f"{round(np.std(metric), 4):.4f}"
        
        # Apply formatting with bb and rs if necessary
        if name in ["F1 score", "NDCG"]:
            mean_str = f"{bb}{mean_str}{rs}"
        
        print(f"{name:>8}: {values_str} | {mean_str}, {std_str}")


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device, batch_size=1000):
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)

    assert n_users == user_Embed_wts.shape[0]
    assert n_items == item_Embed_wts.shape[0]

    # Initialize metrics
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    num_batches = (n_users + batch_size - 1) // batch_size

    # Prepare interaction tensor for the batch
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones(len(train_df), dtype=torch.float32).to(device)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()

    # Collect results across batches
    all_topk_relevance_indices = []
    all_user_ids = []

    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = torch.arange(batch_start, batch_end).to(device)

        # Extract embeddings for the current batch
        user_Embed_wts_batch = user_Embed_wts[batch_user_indices]
        relevance_score_batch = torch.matmul(user_Embed_wts_batch, item_Embed_wts.t())

        # Mask out training user-item interactions from metric computation
        relevance_score_batch = relevance_score_batch * (1 - interactions_t[batch_user_indices])

        # Compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score_batch, K).indices
        all_topk_relevance_indices.append(topk_relevance_indices)
        all_user_ids.extend(batch_user_indices.cpu().numpy())

    # Combine results
    topk_relevance_indices = torch.cat(all_topk_relevance_indices).cpu().numpy()

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()

    # Merge test interactions with top-K predicted relevance indices
    metrics_df = pd.merge(test_interacted_items, pd.DataFrame({'user_id': all_user_ids, 'top_rlvnt_itm': topk_relevance_indices.tolist()}), how='left', on='user_id')

    # Handle missing values and ensure that item_id and top_rlvnt_itm are lists
    metrics_df['item_id'] = metrics_df['item_id'].apply(lambda x: x if isinstance(x, list) else [])
    metrics_df['top_rlvnt_itm'] = metrics_df['top_rlvnt_itm'].apply(lambda x: x if isinstance(x, list) else [])

    # Calculate intersection items
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]

    # Calculate recall, precision, and nDCG
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']) if len(x['item_id']) > 0 else 0, axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    # Generate a binary relevance matrix for test interactions (same as in the first function)
    test_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        length = min(K, len(relevant_items))
        test_matrix[i, :length] = 1

    # Compute IDCG (Ideal DCG)
    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, K + 2)), axis=1)
    
    # Compute DCG based on predicted relevance
    dcg_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        dcg_matrix[i] = [1 if item in relevant_items else 0 for item in predicted_items]
    
    dcg = np.sum(dcg_matrix * (1. / np.log2(np.arange(2, K + 2))), axis=1)

    # Handle cases where idcg == 0 to avoid division by zero
    idcg[idcg == 0.] = 1.

    # Compute nDCG as DCG / IDCG
    ndcg = dcg / idcg

    # Set NaNs in nDCG to zero
    ndcg[np.isnan(ndcg)] = 0.

    # Aggregate metrics
    total_recall = metrics_df['recall'].mean()
    total_precision = metrics_df['precision'].mean()
    total_ndcg = np.mean(ndcg)
    
    return total_recall, total_precision, total_ndcg


def get_metrics_2(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device, batch_size=1000):
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)

    assert n_users == user_Embed_wts.shape[0]
    assert n_items == item_Embed_wts.shape[0]

    # Initialize metrics
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    num_batches = (n_users + batch_size - 1) // batch_size

    # Prepare interaction tensor for the batch
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones(len(train_df), dtype=torch.float32).to(device)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()

    # Collect results across batches
    all_topk_relevance_indices = []
    all_user_ids = []

    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = torch.arange(batch_start, batch_end).to(device)

        # Extract embeddings for the current batch
        user_Embed_wts_batch = user_Embed_wts[batch_user_indices]
        relevance_score_batch = torch.matmul(user_Embed_wts_batch, item_Embed_wts.t())

        # Mask out training user-item interactions from metric computation
        relevance_score_batch = relevance_score_batch * (1 - interactions_t[batch_user_indices])

        # Compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score_batch, K).indices
        topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
        topk_relevance_indices_df['all_user_id'] = topk_relevance_indices_df.index
        topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
        topk_relevance_indices_df = topk_relevance_indices_df[['all_user_id', 'top_rlvnt_itm']]

        # Collect results
        all_topk_relevance_indices.append(topk_relevance_indices_df)
        all_user_ids.extend(batch_user_indices.cpu().numpy())

    # Combine results
    topk_relevance_indices_df_all = pd.concat(all_topk_relevance_indices, ignore_index=True)
    topk_relevance_indices_df_all['all_user_id'] = all_user_ids

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
    
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df_all, how='left', left_on='user_id', right_on='all_user_id')

    # Handle missing values and ensure that item_id and top_rlvnt_itm are lists
    metrics_df['item_id'] = metrics_df['item_id'].apply(lambda x: x if isinstance(x, list) else [])
    metrics_df['top_rlvnt_itm'] = metrics_df['top_rlvnt_itm'].apply(lambda x: x if isinstance(x, list) else [])

    # Calculate intersection items
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]

    # Calculate recall, precision, and nDCG
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']) if len(x['item_id']) > 0 else 0, axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(relevance_scores, k):
        dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(relevance_scores, k) / dcg_max

    metrics_df['ndcg'] = metrics_df.apply(lambda x: ndcg_at_k([1 if i in x['item_id'] else 0 for i in x['top_rlvnt_itm']], K), axis=1)

    # Aggregate metrics
    total_recall = metrics_df['recall'].mean()
    total_precision = metrics_df['precision'].mean()
    total_ndcg = metrics_df['ndcg'].mean()
    
    return total_recall, total_precision, total_ndcg

def get_metrics_unbatched(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device):
        
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)
    
    assert n_users == user_Embed_wts.shape[0]
    assert n_items == item_Embed_wts.shape[0]
    
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones((len(train_df)), dtype=torch.float32).to(device)
    
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
    
    # mask out training user-item interactions from metric computation
    relevance_score = relevance_score * (1 - interactions_t)
    
    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['all_user_id'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['all_user_id', 'top_rlvnt_itm']]
    
    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
    
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id', right_on='all_user_id')
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]
    
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    # Calculate nDCG
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(relevance_scores, k):
        dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(relevance_scores, k) / dcg_max

    metrics_df['ndcg'] = metrics_df.apply(lambda x: ndcg_at_k([1 if i in x['item_id'] else 0 for i in x['top_rlvnt_itm']], K), axis=1)
    
    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean()
    
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, batch_size):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def make_neg_adj_list_2(data, all_items):
    
    all_items = set(all_items)
    
    # Group by user_id and create a list of pos_items
    adj_list = data.groupby('user_id')['item_id'].apply(list).reset_index()

    # Rename the item_id column to pos_items
    adj_list.rename(columns={'item_id': 'pos_items'}, inplace=True)

    # Add the neg_items column
    adj_list['neg_items'] = adj_list['pos_items'].apply(lambda pos: list(all_items - set(pos)))

    neg_adj_list_dict = adj_list.set_index('user_id')[['neg_items']].to_dict(orient='index')
    
    del adj_list
    
    return neg_adj_list_dict

def make_neg_adj_list(data, all_items):
    all_items_set = set(all_items)

    # Group by user_id and aggregate item_ids into lists
    pos_items = data.groupby('user_id')['item_id'].agg(list)
    
    # Compute neg_items by subtracting the pos_items from all_items for each user
    neg_items = pos_items.apply(lambda pos: list(all_items_set.difference(pos)))
    
    # Create a dictionary with user_id as the key and neg_items as the value
    neg_adj_list_dict = pd.Series(neg_items, index=pos_items.index).to_dict()
    
    del pos_items, neg_items, all_items_set
    
    return neg_adj_list_dict

def neg_uniform_sample(train_df, neg_adj_list, n_usr):
    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)

    # Precompute the length of negative item lists
    #neg_lens = np.array([len(neg_adj_list[user]['neg_items']) for user in users])
    neg_lens = np.array([len(neg_adj_list[user]) for user in users])
    
    # Generate random indices for each user
    random_indices = np.random.randint(0, neg_lens)
    
    # Use indices to select negative items for each user
    #neg_items = [neg_adj_list[user]['neg_items'][idx] for user, idx in zip(users, random_indices)]
    neg_items = [neg_adj_list[user][idx] for user, idx in zip(users, random_indices)]
    
    # Add n_usr to each item id in pos_items and neg_items
    pos_items = [item + n_usr for item in pos_items]
    neg_items = [item + n_usr for item in neg_items]
    
    S = np.column_stack((users, pos_items, neg_items))
    
    del users, pos_items, neg_items, random_indices, neg_lens
    
    return S
                 
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
                
def plot_results(plot_name, num_exp, epochs, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    num_test_epochs = len(all_bi_losses[0]['loss'])
    epoch_list = [(j + 1) for j in range(num_test_epochs)]
             
    for i in range(num_exp):
        
        plt.subplot(1, 3, 1)
        # BI Losses
        plt.plot(epoch_list, all_bi_losses[i]['loss'], label=f'Exp {i+1} - BI Total Training Loss', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['bpr_loss'], label=f'Exp {i+1} - BI BPR Training Loss', linestyle='--', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['reg_loss'], label=f'Exp {i+1} - BI Reg Training Loss', linestyle='-.', color='blue')
        
        # KNN Losses
        plt.plot(epoch_list, all_knn_losses[i]['loss'], label=f'Exp {i+1} - KNN Total Training Loss', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['bpr_loss'], label=f'Exp {i+1} - KNN BPR Training Loss', linestyle='--', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['reg_loss'], label=f'Exp {i+1} - KNN Reg Training Loss', linestyle='-.', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        #plt.legend()

        # Plot for metrics
        plt.subplot(1, 3, 2)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['recall'], label=f'Exp {i+1} - BI Recall', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_metrics[i]['precision'], label=f'Exp {i+1} - BI Precision', linestyle='--', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['recall'], label=f'Exp {i+1} - KNN Recall', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_metrics[i]['precision'], label=f'Exp {i+1} - KNN Precision', linestyle='--', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Recall & Precision')
        plt.title('Recall & Precision')
        
        # Plot for metrics
        plt.subplot(1, 3, 3)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['ncdg'], label=f'Exp {i+1} - BI NCDG', linestyle='-', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['ncdg'], label=f'Exp {i+1} - KNN NCDG', linestyle='-', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('NCDG')
        plt.title('NCDG')
        #plt.legend()

    # Custom Legend
    bi_line = mlines.Line2D([], [], color='blue', label='BI')
    knn_line = mlines.Line2D([], [], color='orange', label='KNN')
    plt.legend(handles=[bi_line, knn_line], loc='lower right')
    
    plt.tight_layout()  # Adjust spacing between subplots
    #plt.show()
    
    # Get current date and time
    now = datetime.now()

    # Format date and time as desired (e.g., "2024-08-27_14-30-00")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(plot_name + '_' + timestamp +'.png')  # Save plot to file