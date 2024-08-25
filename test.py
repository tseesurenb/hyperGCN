from sklearn import preprocessing as pp

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import run_experiment, batch_data_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

columns_name=['user_id','item_id','rating','timestamp']
df = pd.read_csv("data/raw/ml-100k/u.data",sep="\t",names=columns_name)
#df = df[df['rating']>=3] # How many ratings are a 3 or above?
print(f'Number of total user-item interactions: {len(df)}')

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()

df['user_id_idx'] = le_user.fit_transform(df['user_id'].values)
df['item_id_idx'] = le_item.fit_transform(df['item_id'].values)

N_USERS = df['user_id_idx'].nunique()
N_ITEMS = df['item_id_idx'].nunique()

interacted_items_df = df.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

print(df.head())
print('-------------------------')
print(interacted_items_df.head())

b = batch_data_loader(df, 1024, N_USERS, N_ITEMS, device)

print(f'{len(b[0])} users: {b[0]}')
print(f'{len(b[1])} pos items: {b[1]}')
print(f'{len(b[2])} neg items: {b[2]}')