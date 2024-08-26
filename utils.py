import torch
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from model import RecSysGNN2, RecSysGNN
from sklearn import preprocessing as pp
from world import config
from data_prep import get_edge_index,  create_jaccard_uuii_adjmat, create_jaccard_uuii_adjmat_coo, create_uuii_adjmat_top_k

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

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device):
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())
    
    # Compute the score of all user-item pairs, including the base embeddings
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
        
    # create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_data['user_id_idx'].values),
        torch.LongTensor(train_data['item_id_idx'].values)
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
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items,topk_relevance_indices_df, how= 'left', left_on = 'user_id_idx',right_on = ['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id_idx']), axis = 1) 
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

    metrics_df['ndcg'] = metrics_df.apply(lambda x: ndcg_at_k([1 if i in x['item_id_idx'] else 0 for i in x['top_rlvnt_itm']], K), axis=1)


    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean()

def get_metrics2(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device):
    # Ensure the embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)
    
    # Get unique test user IDs and move to device
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique()).to(device)
    
    # Compute the relevance scores for all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
    
    # Create sparse tensor for all user-item interactions in the training set
    interaction_indices = torch.stack((
        torch.LongTensor(train_data['user_id_idx'].values),
        torch.LongTensor(train_data['item_id_idx'].values)
    )).to(device)
    interaction_values = torch.ones((len(train_data)), dtype=torch.float32).to(device)
    
    # Convert to dense tensor
    interactions_t = torch.sparse_coo_tensor(interaction_indices, interaction_values, (n_users, n_items)).to_dense()
    
    # Mask out training user-item interactions from relevance scores
    relevance_score = relevance_score * (1 - interactions_t)
    
    # Compute top K scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    
    # Move top K indices to CPU and create DataFrame
    topk_relevance_indices_cpu = topk_relevance_indices.cpu()
    topk_columns = [f'top_indx_{x+1}' for x in range(K)]
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices_cpu.numpy(), columns=topk_columns)
    
    # Add user IDs and create a list of top K relevant items for each user
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[topk_columns].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]
    
    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx', right_on='user_ID')
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df['item_id_idx'], metrics_df['top_rlvnt_itm'])]
    
    # Compute recall and precision
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    return metrics_df['recall'].mean(), metrics_df['precision'].mean()

def batch_data_loader(data, batch_size, n_usr, n_itm, device):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
        
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])

    interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id_idx', right_on = 'users')
    pos_items = interected_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values
    neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device), 
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )
    
def batch_data_loader2(data, batch_size, n_usr, n_itm, device):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    # Reindex users and items
    #unique_users = data['user_id'].unique()
    #unique_items = data['item_id'].unique()

    #user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
    #item_mapping = {item_id: i + n_usr for i, item_id in enumerate(unique_items)}

    #data['user_id_idx'] = data['user_id'].map(user_mapping)
    #data['item_id_idx'] = data['item_id'].map(item_mapping)

    # Group by reindexed user IDs
    interacted_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]

    # Sample users
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()
    users_df = pd.DataFrame(users, columns=['users'])

    # Merge to get positive and negative items
    interacted_items_df = pd.merge(interacted_items_df, users_df, how='right', left_on='user_id_idx', right_on='users')
    pos_items = interacted_items_df['item_id_idx'].apply(lambda x: random.choice(x)).values
    neg_items = interacted_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device),
        torch.LongTensor(list(pos_items)).to(device),
        torch.LongTensor(list(neg_items)).to(device)
    )
    
def train_and_eval(epochs, model, optimizer, train_df, test_df, batch_size, n_users, n_items, train_edge_index, train_edge_attrs, decay, K, device, exp_n, g_seed):
   
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
    pbar.set_description(f'Exp {exp_n:2} | seed {g_seed:2} | # of edges {len(train_edge_index[0])}')
    
    for epoch in pbar:
        n_batch = int(len(train_df)/batch_size)
    
        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []
        
        model.train()
        for batch_idx in range(n_batch):

            optimizer.zero_grad()

            users, pos_items, neg_items = batch_data_loader(train_df, batch_size, n_users, n_items, device)
            users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(users, pos_items, neg_items, train_edge_index, train_edge_attrs)

            bpr_loss, reg_loss = compute_bpr_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
            )
            reg_loss = decay * reg_loss
            final_loss = bpr_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index, train_edge_attrs)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            test_topK_recall,  test_topK_precision, test_ncdg = get_metrics(
                final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K, device
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
    
def plot_loss(num_exp, epochs, light_loss, light_bpr, light_reg, light_recall, light_precision):

    # Plot for losses
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    for i in range(num_exp):
        epoch_list = [(i+1) for i in range(epochs)]
        
        plt.subplot(1, 2, 1)
        
        plt.plot(epoch_list, light_loss, label='Total Training Loss')
        plt.plot(epoch_list, light_bpr, label='BPR Training Loss')
        plt.plot(epoch_list, light_reg, label='Reg Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()

        # Plot for metrics
        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, light_recall, label='Recall')
        plt.plot(epoch_list, light_precision, label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Metrics')
        plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_loss2(num_exp, epochs, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):

    # Plot for losses
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    for i in range(num_exp):
        epoch_list = [(j+1) for j in range(epochs)]
        
        plt.subplot(1, 2, 1)        
        plt.plot(epoch_list, all_bi_losses[i]['loss'], label=f'Exp {i+1} - Total Training Loss')
        plt.plot(epoch_list, all_bi_losses[i]['bpr_loss'], label=f'Exp {i+1} - BPR Training Loss')
        plt.plot(epoch_list, all_bi_losses[i]['reg_loss'], label=f'Exp {i+1} - Reg Training Loss')
        plt.plot(epoch_list, all_knn_losses[i]['loss'], label=f'Exp {i+1} - Total Training Loss')
        plt.plot(epoch_list, all_knn_losses[i]['bpr_loss'], label=f'Exp {i+1} - BPR Training Loss')
        plt.plot(epoch_list, all_knn_losses[i]['reg_loss'], label=f'Exp {i+1} - Reg Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()

        # Plot for metrics
        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, all_bi_metrics[i]['recall'], label=f'Exp {i+1} - Recall')
        plt.plot(epoch_list, all_bi_metrics[i]['precision'], label=f'Exp {i+1} - Precision')
        plt.plot(epoch_list, all_knn_metrics[i]['recall'], label=f'Exp {i+1} - Recall')
        plt.plot(epoch_list, all_knn_metrics[i]['precision'], label=f'Exp {i+1} - Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Metrics')
        plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_loss3(num_exp, epochs, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    for i in range(num_exp):
        epoch_list = [(j + 1) for j in range(epochs)]
        
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
    plt.show()
    
def run_experiment(df, g_seed=42, exp_n = 1, device='cpu', verbose = -1):
    

    train_test_ratio = config['test_ratio']
    train, test = train_test_split(df.values, test_size=train_test_ratio, random_state=g_seed)
    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)

    if verbose >= 1:
        print("train Size: ", len(train_df), " | test size: ", len (test_df))
      
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
    train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)
    
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()

    test_df = test_df[
      (test_df['user_id'].isin(train_user_ids)) & \
      (test_df['item_id'].isin(train_item_ids))
    ]

    test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
    test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)
    
    N_USERS = train_df['user_id_idx'].nunique()
    N_ITEMS = train_df['item_id_idx'].nunique()

    if verbose >= 0:
        print("Number of Unique Users : ", N_USERS)
        print("Number of unique Items : ", N_ITEMS)

    u_t = torch.LongTensor(train_df.user_id_idx)
    i_t = torch.LongTensor(train_df.item_id_idx) + N_USERS

    if verbose >= 1:
        # Verify the ranges
        print("max user index: ", u_t.max().item(), "| min user index: ", u_t.min().item())
        print("max item index: ", i_t.max().item(), "| min item index:", i_t.min().item())
    
    u_t = torch.LongTensor(train_df.user_id_idx)
    i_t = torch.LongTensor(train_df.item_id_idx) + N_USERS

    bi_train_edge_index = torch.stack((
      torch.cat([u_t, i_t]),
      torch.cat([i_t, u_t])
    )).to(device)
         
    #knn_train_adj_df = create_uuii_adjmat2(train_df)
    #knn_train_adj_df = create_pearson_sim_uuii_adjmat(train_df, p_thresh=0.27, j_thresh=0.2)
    #knn_train_adj_df = create_pearson_sim_uuii_adjmat(train_df, p_thresh=config['pears_thresh'], j_thresh=config['jacc_thresh'])
    #knn_train_adj_df = create_jaccard_uuii_adjmat(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_thresh=config['u_sim_thresh'], i_sim_thresh=config['i_sim_thresh'])
    knn_train_adj_df = create_uuii_adjmat_top_k(train_df, u_sim=config['u_sim'], i_sim=config['i_sim'], u_sim_top_k=config['u_sim_top_k'], i_sim_top_k=config['i_sim_top_k']) 
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

    #print(f'LATENT_DIM: {LATENT_DIM} | N_LAYERS: {N_LAYERS} | JACCARD: {config['jacc_thresh']} | PEARSON: {config['pears_thresh']} | BATCH_SIZE: {BATCH_SIZE} | DECAY: {DECAY} | topK: {K} | IS_TEMP: {IS_TEMP} | MODEL: {MODEL}')

    lightgcn = RecSysGNN(
      latent_dim=LATENT_DIM, 
      num_layers=N_LAYERS,
      num_users=N_USERS,
      num_items=N_ITEMS,
      model=MODEL,
      is_temp=IS_TEMP,
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

    if verbose >= 1:
        plot_loss(EPOCHS, losses['loss'],  losses['bpr_loss'],  losses['reg_loss'], metrics['recall'], metrics['precision'])
        
    
    return losses, metrics