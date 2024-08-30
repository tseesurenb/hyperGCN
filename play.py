'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import torch
import torch.backends
import torch.mps
import numpy as np
from utils import run_experiment, plot_loss3
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
df, u_df, i_df, stats = dp.load_data(dataset = config['dataset'], u_min_interaction_threshold = 10, i_min_interaction_threshold = 10, verbose=config['verbose'])
df = df[df['rating']>=3] # How many ratings are a 3 or above?

print(df.head())