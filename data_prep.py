'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import similarity_func as sim
from world import config
import gc

import os
from scipy.sparse import coo_matrix, vstack, hstack, save_npz, load_npz

from scipy.sparse import coo_matrix, vstack, hstack

#import dask.dataframe as dd

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def get_edge_index(sparse_matrix):    
    # Extract row, column indices and data values
    row_indices = sparse_matrix.row
    column_indices = sparse_matrix.col
    data = sparse_matrix.data
    
    # Prepare edge index
    edge_index = np.vstack((row_indices, column_indices))
    
    del row_indices, column_indices
    
    return edge_index, data

def create_uuii_adjmat(df, u_sim='cosine', i_sim='jaccard', u_sim_top_k=20, i_sim_top_k=20, self_sim=False, verbose=-1):
    
    #file_path=f"pre_proc/{config['dataset']}_u_{u_sim}_{u_sim_top_k}_i_{i_sim}_{i_sim_top_k}_self_{self_sim}_uuii_adjmat.npz"
    
    # Check if the file exists
    #if os.path.exists(file_path):
    #    if verbose > 0:
    #        print('Loading adjacency matrix from file...')
    #    # Load the sparse matrix from the file
    #    combined_adjacency = load_npz(file_path)
    #    return combined_adjacency

    if verbose > 0:
        print('Creating user-item matrix...')
    # Convert to NumPy arrays
    user_ids = df['user_id'].to_numpy()
    item_ids = df['item_id'].to_numpy()

    # Create a sparse matrix directly
    user_item_matrix_coo = coo_matrix((np.ones(len(df)), (user_ids, item_ids)))
    user_item_matrix = user_item_matrix_coo.toarray()

    if verbose > 0:
        print('User-item coo matrix created.')
        
    # Calculate user-user similarity matrix
    if u_sim == 'cosine':
        user_user_sim_matrix = sim.cosine_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim, verbose=verbose)
    elif u_sim == 'mix':
        user_user_sim_matrix = sim.fusion_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim, verbose=verbose)
    else:
        user_user_sim_matrix = sim.jaccard_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim, verbose=verbose)
        
    if verbose > 0:
        print('User-User Sim matrix created.')
    
    # Calculate item-item similarity matrix
    if i_sim == 'cosine':
        item_item_sim_matrix = sim.cosine_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim, verbose=verbose)
    elif i_sim == 'mix':
        item_item_sim_matrix = sim.fusion_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim, verbose=verbose)
    else:
        item_item_sim_matrix = sim.jaccard_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim, verbose=verbose)
        
    if verbose > 0:
        print('Item-Item Sim matrix created.')
    
    # Stack user-user and item-item matrices vertically and horizontally
    num_users = user_user_sim_matrix.shape[0]
    num_items = item_item_sim_matrix.shape[0]

    # Initialize combined sparse matrix
    combined_adjacency = vstack([
        hstack([user_user_sim_matrix, coo_matrix((num_users, num_items))]),
        hstack([coo_matrix((num_items, num_users)), item_item_sim_matrix])
    ])

    if verbose > 0:
        print('User-item and item-item adjacency matrices created.')
    
    # Save the sparse matrix to a file
    #save_npz(file_path, combined_adjacency)
    
    del user_item_matrix_coo, user_item_matrix, user_user_sim_matrix, item_item_sim_matrix

    return combined_adjacency

def load_data(dataset = "ml-100k", u_min_interaction_threshold = 20, i_min_interaction_threshold = 20, verbose = 0):
    
    user_df = None
    item_df = None
    df_selected = None
    ratings_df = None
    rating_stat = None
    df = None
        
    if dataset == 'ml-100k':
        # Paths for ML-100k data files
        ratings_path = f'data/{dataset}/u.data'
        movies_path = f'data/{dataset}/u.item'
        users_path = f'data/{dataset}/u.user'
        
        print(f'Loading data for {dataset} from {ratings_path} ...')
        
        
        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='\t', names=["user_id", "item_id", "rating", "timestamp"])

        # Load the entire movie dataframe into memory
        genre_columns = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        item_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_columns)
        
        # Create the genres column by concatenating genre names where the value is 1
        item_df['genres'] = item_df[genre_columns].apply(lambda row: '|'.join([genre for genre, val in row.items() if val == 1]), axis=1)
    
        # Keep only the necessary columns
        item_df = item_df[["item_id", "title", "genres"]]
        item_df = item_df.set_index('item_id')
        
        # Load the entire user dataframe into memory 1|24|M|technician|85711
        user_df = pd.read_csv(users_path, sep='|', encoding='latin-1', names=["user_id", "age_group", "sex", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('user_id')
    
    elif dataset == 'ml-1m':
        # Paths for ML-1M data files
        ratings_path = f'data/{dataset}/ratings.dat'
        movies_path = f'data/{dataset}/movies.dat'
        users_path = f'data/{dataset}/users.dat'

        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='::', names=["user_id", "item_id", "rating", "timestamp"], engine='python', encoding='latin-1')
        
        # Load the entire movie dataframe into memory
        item_df = pd.read_csv(movies_path, sep='::', names=["item_id", "title", "genres"], engine='python', encoding='latin-1')
        item_df = item_df.set_index('item_id')
        
        # Load the entire user dataframe into memory UserID::Gender::Age::Occupation::Zip-code -> 1::F::1::10::48067 
        user_df = pd.read_csv(users_path, sep='::', encoding='latin-1', names=["user_id", "sex", "age_group", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('user_id')
        
    elif dataset == 'amazon_cloth':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/df_modcloth.csv'

        # Load the entire ratings dataframe into memory        
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['item_id', 'user_id', 'rating', 'timestamp']]
        
        # Parse timestamps and remove timezone information if present        
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp'], errors='coerce')
        # Convert to Unix time in seconds
        df_selected['timestamp'] = df_selected['timestamp'].astype(int) // 10**9
    
    elif dataset == 'amazon_fashion':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/amazon_fashion.csv'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]
                        
    elif dataset == 'amazon_book':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/books.csv'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        df_selected = df[['user_id', 'asin', 'rating', 'timestamp']]

        # Rename the columns
        df_selected.columns = ['user_id', 'item_id', 'rating', 'timestamp']
                
    elif dataset == 'epinion':
        # Paths for ML-1M data files
        ratings_path = f'data/epinion/rating_with_timestamp.txt'
        
        # Read the text file into a DataFrame
        df = pd.read_csv('data/epinion/rating_with_timestamp.txt', sep=r'\s+', header=None)

        # Assign column names
        df.columns = ['user_id', 'item_id', 'category_id', 'rating', 'helpfulness', 'timestamp']

        # Select only the columns we need
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]
    
    elif dataset == 'douban_book':
        ratings_path = f'data/douban/bookreviews_cleaned.txt'
            
        # Read the text file into a DataFrame
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'book_id', 'rating', 'time']]
        df_selected = df.rename(columns={'user_id': 'user_id', 'book_id': 'item_id', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp']).astype(int) // 10**9
        
    elif dataset == 'douban_music':
        # Read the text file into a DataFrame
        ratings_path = f'data/douban/musicreviews_cleaned.txt'
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'music_id', 'rating', 'time']]
        df_selected = df.rename(columns={'user_id': 'user_id', 'music_id': 'item_id', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp']).astype(int) // 10**9
        
    elif dataset == 'douban_movie':
        # Read the text file into a DataFrame
        ratings_path = f'data/douban/moviereviews_cleaned.txt'
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'movie_id', 'rating', 'time']]
        df_selected = df.rename(columns={'user_id': 'user_id', 'movie_id': 'item_id', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp']).astype(int) // 10**9
    
    elif dataset == 'yelp':
        # Read the text file into a DataFrame
        ratings_path = f'data/yelp/yelp_reviews.csv'
        df = pd.read_csv(ratings_path, sep=',')

        # Select and rename the columns to match the desired format
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]
                
        # Convert the timestamp column to Unix timestamps
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp']).astype(int) // 10**9
        
    elif dataset == 'gowalla':
        # Paths for ML-1M data files
        ratings_path = f'data/gowalla/gowalla.csv'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(ratings_path, header=0)
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9

        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]
              
    else:
        print(f'{br}No data is loaded for dataset: {dataset} !!!{rs}')

    if df_selected is not None:
    
        #_lbl_user = preprocessing.LabelEncoder()
        #_lbl_movie = preprocessing.LabelEncoder()
        
        #ratings_df.userId = _lbl_user.fit_transform(ratings_df.userId.values)
        #ratings_df.itemId = _lbl_movie.fit_transform(ratings_df.itemId.values)
        
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df_selected['user_id'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= u_min_interaction_threshold].index
        df_user_filtered = df_selected[df_selected['user_id'].isin(filtered_users)]
        
        # Filter items with at least a minimum number of interactions
        item_interaction_counts = df_user_filtered['item_id'].value_counts()
        filtered_items = item_interaction_counts[item_interaction_counts >= i_min_interaction_threshold].index
        df_filtered = df_user_filtered[df_user_filtered['item_id'].isin(filtered_items)]
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
            
        num_users = len(ratings_df['user_id'].unique())
        num_items = len(ratings_df['item_id'].unique())
        mean_rating = round(ratings_df['rating'].mean(), 2)
        num_ratings = len(ratings_df)
        
        # Calculate the max-min time distance
        min_timestamp = ratings_df['timestamp'].min()
        max_timestamp = ratings_df['timestamp'].max()
        max_min_time_distance = round((max_timestamp - min_timestamp) / 86400, 0)
        
        rating_stat = {'num_users': num_users, 'num_items': num_items, 'mean_rating': mean_rating, 'num_interactions': num_ratings, 'time_distance': max_min_time_distance}

        if verbose > -1:
            print(f'{br}{dataset}{rs} | {rating_stat}')
        elif verbose == 1:
            print(ratings_df.head())
            
        # Clear memory of large DataFrames that are no longer needed
        del df
        del df_filtered
        del df_selected
        gc.collect()
        
    else:
        print(f'{br}No data is loaded for dataset: {dataset} !!! {rs}')
        
    return ratings_df, user_df, item_df, rating_stat

def load_data_from_adj_list(dataset = "gowalla_2", verbose = 0):
    
    train_df = None
    test_df = None
    df = None
                      
    if dataset == 'gowalla_2':
        # Paths for ML-1M data files
        train_path = f'data/gowalla/train_coo.txt'
        test_path = f'data/gowalla/test_coo.txt'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(train_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        train_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
                              
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(test_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        test_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
        
        # Explicitly delete the dataframe and run garbage collection
    
    elif dataset == 'yelp2018':
        # Paths for ML-1M data files
        train_path = f'data/yelp2018/train_coo.txt'
        test_path = f'data/yelp2018/test_coo.txt'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(train_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        train_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
                      
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(test_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        test_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
        
    elif dataset == 'amazon-book':
        # Paths for ML-1M data files
        train_path = f'data/amazon-book/train_coo.txt'
        test_path = f'data/amazon-book/test_coo.txt'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(train_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        train_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
                      
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(test_path, header=0, sep=' ')
        # Select the relevant columns 'asin', 'user_id', 'rating', 'timestamp'
        test_df = df[['user_id', 'item_id', 'rating', 'timestamp']]
                              
    else:
        print(f'{br}No data is loaded for dataset: {dataset} !!!{rs}')
        
    del df
    gc.collect()
        
    return train_df, test_df