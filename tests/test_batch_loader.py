import pandas as pd
import polars as pl
import numpy as np
import torch
import random
import time
import timeit

# Define the batch_data_loader function (copy the function code here)

seed = 42
# Set seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def batch_data_loader(data, batch_size, n_usr, n_itm, device):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interacted_items_df = data.groupby('user_id')['item_id'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]
    
    #print(len(interacted_items_df))

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
        
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])
    
    #print(len(users_df))
    
    interacted_items_df = pd.merge(interacted_items_df, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
    
    #print(len(interacted_items_df))
    
    # Vectorize positive item sampling
    pos_items = interacted_items_df['item_id'].apply(lambda x: np.random.choice(x)).values
    
    #print(pos_items + n_usr)

    # Vectorize negative item sampling
    neg_items = interacted_items_df['item_id'].apply(lambda x: sample_neg(x)).values
    
    #print(neg_items + n_usr)

    return (
        torch.LongTensor(list(users)).to(device), 
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )

def batch_data_loader_faster(adj_list, batch_size, n_usr, n_itm, indices, device):

    indices = [x for x in range(n_usr)]
    
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
        
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])
    
    items_df = pd.merge(adj_list, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
    #items_df = adj_list[adj_list['user_id'].isin(users)]
    
    
    # Vectorize positive item sampling
    pos_items = items_df['pos_items'].apply(lambda x: np.random.choice(x)).values

    # Vectorize negative item sampling
    neg_items = items_df['neg_items'].apply(lambda x: np.random.choice(x)).values
    
    return (
        torch.LongTensor(list(users)).to(device), 
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )
    
def batch_data_loader_optimized(adj_list, batch_size, n_usr, n_itm, device):

    indices = np.arange(n_usr)
    
    if n_usr < batch_size:
        users = np.random.choice(indices, batch_size, replace=True)
    else:
        users = np.random.choice(indices, batch_size, replace=False)
        
    users.sort()
    items_df = adj_list[adj_list['user_id'].isin(users)]
    
    # Efficient positive and negative item sampling
    pos_items = np.array([np.random.choice(pos) for pos in items_df['pos_items']])
    neg_items = np.array([np.random.choice(neg) for neg in items_df['neg_items']])
    
    return (
        torch.LongTensor(users).to(device), 
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )
    
def batch_data_loader_GPT(adj_list, batch_size, n_usr, n_itm, device):

    indices = np.arange(n_usr)
    
    if n_usr < batch_size:
        users = np.random.choice(indices, batch_size, replace=True)
    else:
        users = np.random.choice(indices, batch_size, replace=False)
        
    users.sort()
    
    # Efficiently filter the DataFrame using boolean indexing
    items_df = adj_list[adj_list['user_id'].isin(users)]
    
    # Efficient positive and negative item sampling
    pos_items = np.array([np.random.choice(pos) for pos in items_df['pos_items'].to_numpy()])
    neg_items = np.array([np.random.choice(neg) for neg in items_df['neg_items'].to_numpy()])
    
    return (
        torch.LongTensor(users).to(device), 
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )


def preprocess_data(data, n_usr, n_itm):
    # Create a list of positive item lists per user
    user_item_dict = {}
    for user_id in range(n_usr):
        user_items = data[data[:, 0] == user_id][:, 1]
        user_item_dict[user_id] = user_items
    
    all_items = set(range(n_itm))
    # Precompute negative items
    neg_items_dict = {}
    for user_id, pos_items in user_item_dict.items():
        neg_items_dict[user_id] = list(all_items - set(pos_items))
    
    return user_item_dict, neg_items_dict


def batch_data_loader_optimized_GPT_2(user_item_dict, neg_items_dict, batch_size, n_usr, n_itm, device):
    indices = np.arange(n_usr)
    
    if n_usr < batch_size:
        users = np.random.choice(indices, batch_size, replace=True)
    else:
        users = np.random.choice(indices, batch_size, replace=False)
    
    users.sort()
    
    pos_items = np.array([np.random.choice(user_item_dict[user]) for user in users])
    neg_items = np.array([np.random.choice(neg_items_dict[user]) for user in users])
    
    return (
        torch.LongTensor(users).to(device), 
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )
    
def batch_data_loader_optimized_GPT_3(user_item_dict, neg_items_dict, batch_size, n_usr, n_itm, device):
    indices = np.arange(n_usr)
    
    if n_usr < batch_size:
        users = np.random.choice(indices, batch_size, replace=True)
    else:
        users = np.random.choice(indices, batch_size, replace=False)
    
    users.sort()
    
    # Efficiently retrieve positive and negative items
    pos_items = np.array([np.random.choice(user_item_dict[user]) for user in users])
    neg_items = np.array([np.random.choice(neg_items_dict[user]) for user in users])
    
    return (
        torch.LongTensor(users).to(device), 
        torch.LongTensor(pos_items).to(device) + n_usr,
        torch.LongTensor(neg_items).to(device) + n_usr
    )


def pd_create_adj_list(data):
    # Set of all items
    all_items = set(data['item_id'].unique())

    # Group by user_id and create a list of pos_items
    adj_list = data.groupby('user_id')['item_id'].apply(list).reset_index()

    # Rename the item_id column to pos_items
    adj_list.rename(columns={'item_id': 'pos_items'}, inplace=True)

    # Add the neg_items column
    adj_list['neg_items'] = adj_list['pos_items'].apply(lambda pos: list(all_items - set(pos)))

    return adj_list

def pl_create_adj_list(df: pl.DataFrame):
    # Create a set of all unique items
    set_items = set(df['item_id'].unique().to_list())
    
    all_items = pl.lit(list(set_items))
    
    adj_list = df.group_by('user_id', maintain_order=True).agg(pl.col('item_id'))

    # Rename the item_id column to pos_items
    adj_list = adj_list.rename({"item_id": "pos_items"})
    
    # Add the neg_items column
    adj_list = adj_list.with_columns(
            neg_items=all_items.list.set_difference(pl.col("pos_items"))
        )
        
    return adj_list


def test_pd_funcs(pd_data):
    pd_adj_list = pd_create_adj_list(pd_data)

    # Test the first batch_data_loader function
    t1 = time.time()
    users, pos_items, neg_items = batch_data_loader(pd_data, batch_size, n_usr, n_itm, device)
    t2 = time.time()

    p_text = "Time taken by original batch_data_loader:"
    print(f"{p_text:>45}{t2 - t1:.5f}")

    # Test the first batch_data_loader function
    t1 = time.time()
    users, pos_items, neg_items = batch_data_loader_optimized(pd_adj_list, batch_size, n_usr, n_itm, device)
    t2 = time.time()

    p_text = "Time taken by optimized batch_data_loader:"
    print(f"{p_text:>45}{t2 - t1:.5f}")


    indices = [x for x in range(n_usr)]
    # Test the first batch_data_loader function
    t1 = time.time()
    users, pos_items, neg_items = batch_data_loader_faster(pd_adj_list, batch_size, n_usr, n_itm, indices, device)
    t2 = time.time()

    p_text = "Time taken by faster batch_data_loader:"
    print(f"{p_text:>45}{t2 - t1:.5f}")

    t1 = time.time()
    users, pos_items, neg_items = batch_data_loader_GPT(pd_adj_list, batch_size, n_usr, n_itm, device) #(adj_list, batch_size, n_usr, n_itm, device)
    t2 = time.time()

    p_text = "Time taken by GPT batch_data_loader:"
    print(f"{p_text:>45}{t2 - t1:.5f}");


big = True
# Create a larger sample dataset for testing
if big:
    n_usr = 30000
    n_itm = 50000
    data_size = 5000000  # Number of interactions
else:
    # Create a larger sample dataset for testing
    n_usr = 3   
    n_itm = 5
    data_size = 15  # Number of interactions

batch_size = 1024  # You can adjust this to a suitable value for your test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


pl = False

if pl:
    pl_data = pl.DataFrame({
        'user_id': np.random.randint(0, n_usr, data_size),
        'item_id': np.random.randint(0, n_itm, data_size)
    })

    pl_adj_list = pl_create_adj_list(pl_data)
    print(pl_adj_list.head())
else:
    pd_data = pd.DataFrame({
        'user_id': np.random.randint(0, n_usr, data_size),
        'item_id': np.random.randint(0, n_itm, data_size)
    })

    test_pd_funcs(pd_data)