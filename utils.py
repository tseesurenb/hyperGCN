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

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"


def print_metrics(recalls, precs, f1s, ncdg, stats): 
    
    print(f" Dataset: {config['dataset']}, num_users: {stats['num_users']}, num_items: {stats['num_items']}, num_interactions: {stats['num_interactions']}")
    
    if config['edge'] == 'bi':
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs}")
    else:
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | SIM (mode-{config['weight_mode']}, self-{config['self_sim']}): {br}u-{config['u_sim']}(topK {config['u_sim_top_k']}), i-{config['i_sim']}(topK {config['i_sim_top_k']}){rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs}")

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

def get_metrics_new(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device):
    test_user_ids = torch.LongTensor(test_data['user_id'].unique()).to(device)

    # Compute the score of all user-item pairs in chunks to avoid large memory allocation
    chunk_size = 5000  # Adjust this based on available GPU memory
    topk_relevance_indices = []
    
    for i in range(0, n_users, chunk_size):
        user_chunk = user_Embed_wts[i:i+chunk_size]
        relevance_score_chunk = torch.matmul(user_chunk, item_Embed_wts.T)  # User-item relevance score matrix
        
        # Create sparse tensor of user-item interactions for this chunk
        chunk_user_ids = torch.arange(i, min(i + chunk_size, n_users)).to(device)
        user_interactions = train_data[train_data['user_id'].isin(chunk_user_ids.cpu())]
        
        i_chunk = torch.stack((
            torch.LongTensor(user_interactions['user_id'].values) - i,
            torch.LongTensor(user_interactions['item_id'].values)
        ))
        v_chunk = torch.ones(len(user_interactions), dtype=torch.float32)

        interactions_t_chunk = torch.sparse_coo_tensor(i_chunk, v_chunk, (len(chunk_user_ids), n_items)).to(device)
        
        # Mask out training user-item interactions
        relevance_score_chunk = relevance_score_chunk * (1 - interactions_t_chunk.to_dense())
        
        # Compute top K items for each user in this chunk
        topk_indices_chunk = torch.topk(relevance_score_chunk, K, dim=1).indices.cpu()
        topk_relevance_indices.append(topk_indices_chunk)
    
    # Combine top K indices from all chunks
    topk_relevance_indices = torch.cat(topk_relevance_indices, dim=0)

    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.numpy(), columns=['top_indx_' + str(x + 1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id')['item_id'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id', right_on='user_ID')
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


def get_metrics_orig(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_data, K, device):
    test_user_ids = torch.LongTensor(test_data['user_id'].unique())
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts,0, 1))

    # create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    ))
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))\
        .to_dense().to(device)
    
    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    # measure overlap between recommended (top-scoring) and held-out user-item 
    # interactions
    test_interacted_items = test_data.groupby('user_id')['item_id'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items,topk_relevance_indices_df, how= 'left', left_on = 'user_id',right_on = ['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id']), axis = 1) 
    metrics_df['precision'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/K, axis = 1)
  
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
        
def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device):
    test_user_ids = torch.LongTensor(test_data['user_id'].unique())
    
    # Compute the score of all user-item pairs, including the base embeddings
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
        
    # create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_data['user_id'].values),
        torch.LongTensor(train_data['item_id'].values)
    ))
    v = torch.ones((len(train_data)), dtype=torch.float32)
    
    #interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))\
    #    .to_dense().to(device)
    
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items))\
        .to_dense().to(device)
    
    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))
    
    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    
    #topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
    
    topk_relevance_indices_cpu = topk_relevance_indices.cpu()
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices_cpu.numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
    
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    # measure overlap between recommended (top-scoring) and held-out user-item 
    # interactions
    test_interacted_items = test_data.groupby('user_id')['item_id'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items,topk_relevance_indices_df, how= 'left', left_on = 'user_id',right_on = ['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id']), axis = 1) 
    metrics_df['precision'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/K, axis = 1)
    
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

def batch_data_loader_by_adj_list(adj_list, batch_size, n_usr, n_itm, device):

    indices = np.arange(n_usr)
    
    if n_usr < batch_size:
        users = np.random.choice(indices, batch_size, replace=True)
    else:
        users = np.random.choice(indices, batch_size, replace=False)
        
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])
    
    # Efficiently filter the DataFrame using boolean indexing
    #items_df = adj_list[adj_list['user_id'].isin(users)]
    items_df = pd.merge(adj_list, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
    
    # Efficient positive and negative item sampling
    pos_items = np.array([np.random.choice(pos) for pos in items_df['pos_items'].to_numpy()])
    neg_items = np.array([np.random.choice(neg) for neg in items_df['neg_items'].to_numpy()])
     
    return (
        torch.LongTensor(users).to(device), 
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )
    
def make_adj_list(data):
    # Set of all items
    all_items = set(data['item_id'].unique())

    # Group by user_id and create a list of pos_items
    adj_list = data.groupby('user_id')['item_id'].apply(list).reset_index()

    # Rename the item_id column to pos_items
    adj_list.rename(columns={'item_id': 'pos_items'}, inplace=True)

    # Add the neg_items column
    adj_list['neg_items'] = adj_list['pos_items'].apply(lambda pos: list(all_items - set(pos)))

    # Convert adj_list DataFrame to a dictionary
    adj_list_dict = adj_list.set_index('user_id')['pos_items'].to_dict()

    return adj_list_dict

def UniformSample_using_interaction_list(adj_list, train_df):
    """
    Sampling function based on the existing interaction list.
    :param adj_list: A dictionary where each key is a user and the value is a list of items the user has interacted with.
    :param train_df: A DataFrame containing user-item interactions. Columns: ['user', 'item']
    :return:
        np.array: A numpy array of triplets [user, positem, negitem]
    """
    # Convert train_df to numpy array for efficient processing
    interactions = train_df.to_numpy()
    
    S = []
    
    #print("interactions: ", interactions)
    for i, (user, positem, _, _) in enumerate(interactions):
        
        # Get the list of positive items for the user
        
        posForUser = adj_list[user]
        
        if len(posForUser) == 0:
            continue
             
        # Directly use positem from the interaction list
        while True:
            # Sample a negative item that the user has not interacted with
            negitem = np.random.randint(0, len(adj_list))
            if negitem in posForUser:
                continue
            else:
                break
        
        S.append([user, positem, negitem])
        
    return np.array(S)

def batch_data_loader(data, batch_size, n_usr, n_itm, device):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interacted_items_df = data.groupby('user_id')['item_id'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
        
    #users = np.random.choice(n_usr, batch_size, replace=n_usr < batch_size)
       
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])
    
    interacted_items_df = pd.merge(interacted_items_df, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
    #pos_items = interacted_items_df['item_id'].apply(lambda x : random.choice(x)).values
    #neg_items = interacted_items_df['item_id'].apply(lambda x: sample_neg(x)).values
    
    # Vectorize positive item sampling
    pos_items = interacted_items_df['item_id'].apply(lambda x: np.random.choice(x)).values

    # Vectorize negative item sampling
    neg_items = interacted_items_df['item_id'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device), 
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )
    
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
            
def train_and_eval(epochs, model, optimizer, train_df, train_adj_list, test_df, test_adj_list, batch_size, n_users, n_items, train_edge_index, train_edge_attrs, decay, topK, device, exp_n, g_seed):
   
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

    n_batch = len(train_df) // batch_size + 1
    
    pbar = tqdm(range(epochs), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    #pbar.set_description(f'Exp {exp_n:2} | seed {g_seed:2} | #edges {len(train_edge_index[0]):6}')
    
    for epoch in pbar:
    
        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []
        
        #users, pos_items, neg_items = batch_data_loader_by_adj_list(train_adj_list, batch_size, n_users, n_items, device)
        S = UniformSample_using_interaction_list(train_adj_list, train_df)
        
        users = torch.Tensor(S[:, 0]).long()
        pos_items = torch.Tensor(S[:, 1]).long()
        neg_items = torch.Tensor(S[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        model.train()
        for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(minibatch(users,
                                             pos_items,
                                             neg_items,
                                             batch_size=batch_size)):

                optimizer.zero_grad()
                
                #for batch_idx in range(n_batch):
            
                    # Start the timer
            
                    #users, pos_items, neg_items = batch_data_loader(train_df, batch_size, n_users, n_items, device)
                    #batch_data_loader_by_adj_list(adj_list, batch_size, n_usr, n_itm, device):
                    #users, pos_items, neg_items = batch_data_loader_by_adj_list(train_adj_list, batch_size, n_users, n_items, device)
            
                users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(batch_users, batch_pos, batch_neg, train_edge_index, train_edge_attrs)
                
                bpr_loss, reg_loss = compute_bpr_loss(
                    batch_users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
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
            
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, out = model(train_edge_index, train_edge_attrs)
                final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
                test_topK_recall,  test_topK_precision, test_ncdg = get_metrics(
                    final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, topK, device
                )
            
            f1 = (2 * test_topK_recall * test_topK_precision) / (test_topK_recall + test_topK_precision)
                
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
    
def plot_results(plot_name, num_exp, epochs, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    num_test_epochs = len(all_bi_losses[0]['loss'])
        
    for i in range(num_exp):
        epoch_list = [(j + 1) for j in range(num_test_epochs)]
        
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



def run_experiment(df, g_seed=42, exp_n = 1, device='cpu', verbose = -1):

    train_test_ratio = config['test_ratio']
    train, test = train_test_split(df.values, test_size=train_test_ratio, random_state=g_seed)
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)

    if verbose >= 1:
        print("train Size: ", len(train_df), " | test size: ", len (test_df))
    
    # Step 1: Make sure that the user and item pairs in the test set are also in the training set
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()

    test_df = test_df[
      (test_df['user_id'].isin(train_user_ids)) & \
      (test_df['item_id'].isin(train_item_ids))
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

    if verbose >= 0:
        print("Number of unique Users : ", N_USERS)
        print("Number of unique Items : ", N_ITEMS)

    train_adj_list = make_adj_list(train_df)
    test_adj_list = make_adj_list(test_df)
    
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
      weight_mode = config['weight_mode']
    )
    gcn_model.to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=LR)

    losses, metrics = train_and_eval(EPOCHS, 
                                     gcn_model, 
                                     optimizer, 
                                     train_df,
                                     train_adj_list,
                                     test_df,
                                     test_adj_list,
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

    N_USERS = train_df['user_id'].nunique()
    N_ITEMS = train_df['item_id'].nunique()
    
    train_adj_list = make_adj_list(train_df)
    test_adj_list = make_adj_list(test_df)

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
         
    #knn_train_adj_df = create_uuii_adjmat_by_threshold(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_thresh=config['u_sim_thresh'], i_sim_thresh=config['i_sim_thresh'], self_sim=config['self_sim'])
    knn_train_adj_df = create_uuii_adjmat(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_top_k=config['u_sim_top_k'], i_sim_top_k=config['i_sim_top_k'], self_sim=config['self_sim']) 
    knn_train_edge_index, train_edge_attrs = get_edge_index(knn_train_adj_df)
    
    #print(len(knn_train_edge_index[0]))
    #print(len(train_edge_attrs))

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
    
    #assert train_edge_index.max().item() < (N_USERS + N_ITEMS), "Index out of bounds"
    #assert train_edge_index.min().item() >= 0, "Negative index found"
    
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

    #train_and_eval(epochs, model, optimizer, train_df, test_df, batch_size, n_users, n_items, train_edge_index, decay, K)
    losses, metrics = train_and_eval(EPOCHS, 
                                     lightgcn, 
                                     optimizer, 
                                     train_df,
                                     train_adj_list,
                                     test_df, 
                                     test_adj_list,
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