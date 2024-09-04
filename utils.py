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
        
# batched version of the metrics
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
    i = 0

    for batch_start in range(0, n_users, batch_size):
        i += 1
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = torch.arange(batch_start, batch_end).to(device)

        # Extract embeddings for the current batch
        user_Embed_wts_batch = user_Embed_wts[batch_user_indices]
        relevance_score_batch = torch.matmul(user_Embed_wts_batch, item_Embed_wts.t())

        # Prepare interaction tensor for the batch
        i = torch.stack((
            torch.LongTensor(train_df['user_id'].values),
            torch.LongTensor(train_df['item_id'].values)
        )).to(device)
        
        v = torch.ones(len(train_df), dtype=torch.float32).to(device)
        interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()

        interactions_t_cpu = interactions_t.to(device)

        # Mask out training user-item interactions from metric computation
        relevance_score_batch = relevance_score_batch * (1 - interactions_t_cpu[batch_user_indices])

        # Compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score_batch, K).indices
        topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
        topk_relevance_indices_df['all_user_id'] = topk_relevance_indices_df.index
        topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
        topk_relevance_indices_df = topk_relevance_indices_df[['all_user_id', 'top_rlvnt_itm']]

        # Measure overlap between recommended (top-scoring) and held-out user-item interactions
        test_interacted_items = test_df[test_df['user_id'].isin(batch_user_indices.cpu().numpy())]
        test_interacted_items = test_interacted_items.groupby('user_id')['item_id'].apply(list).reset_index()

        metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id', right_on='all_user_id')
        
        print(f"\nMetrics DataFrame with Intersection Items - {batch_start},{i}/{num_batches}:\n {metrics_df}")
        
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

        # Aggregate metrics
        total_recall += metrics_df['recall'].mean() * len(batch_user_indices)
        total_precision += metrics_df['precision'].mean() * len(batch_user_indices)
        total_ndcg += metrics_df['ndcg'].mean() * len(batch_user_indices)

    # Calculate overall metrics
    num_users = n_users
    
    return total_recall / num_users, total_precision / num_users, total_ndcg / num_users

def get_metrics_old(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device):
        
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
    
    #print("\nMetrics DataFrame with Intersection Items:\n", metrics_df)
    
    #sys.exit()
    
    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean()


def get_metrics_lightGCN(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device):
        
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)
    
    assert n_users == user_Embed_wts.shape[0]
    assert n_items == item_Embed_wts.shape[0]
    
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
    #print("Relevance Score:\n", relevance_score)

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
    topk_relevance_indices_np = topk_relevance_indices.cpu().numpy()
    
    #topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
    #topk_relevance_indices_df['all_user_id'] = topk_relevance_indices_df.index
    #topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    #topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]
    
    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
    
    #metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id', right_on='all_user_id')
    #metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]
    
    
     # Compute precision and recall
    recall_precision_results = RecallPrecision_ATk(
        test_interacted_items['item_id'].tolist(),
        topk_relevance_indices_np,
        K
    )
    
    ncdg = NDCGatK_r(
        test_interacted_items['item_id'].tolist(),
        topk_relevance_indices_np,
        K
    )
    
    recall = recall_precision_results['recall']
    precision = recall_precision_results['precision']
    
    recall = recall / n_users
    precision = precision / n_users
    ncdg = ncdg / n_users
    
    #metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']), axis=1)
    #metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    #metrics_df['ndcg'] = metrics_df.apply(lambda x: ndcg_at_k([1 if i in x['item_id'] else 0 for i in x['top_rlvnt_itm']], K), axis=1)
    
    #print("Metrics DataFrame with Intersection Items:\n", metrics_df)
    
    #sys.exit()
    
    return recall, precision, ncdg

    #return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean()
    
    
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
          

def data_loader(train_df, batch_size, n_usr, n_itm, device):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = train_df.groupby('user_id')['item_id'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])

    interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
    pos_items = interected_items_df['item_id'].apply(lambda x : random.choice(x)).values
    neg_items = interected_items_df['item_id'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device),
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )
       
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

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')