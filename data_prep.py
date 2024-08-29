'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

#import dask.dataframe as dd

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def get_edge_index(matrix_df):
    # Convert the DataFrame to a numpy array
    dense_matrix = matrix_df.values
    
    # Convert to sparse COO matrix
    sparse_matrix = coo_matrix(dense_matrix)
    
    # Extract row, column indices and data values
    row_indices = sparse_matrix.row
    column_indices = sparse_matrix.col
    data = sparse_matrix.data
    
    # Prepare edge index
    edge_index = np.vstack((row_indices, column_indices))
    
    return edge_index, data

def jaccard_similarity_by_threshold(matrix, threshold=0.0, self_sim=False):
    
    binary_matrix = (matrix > 0).astype(int)
    
    intersection = np.dot(binary_matrix, binary_matrix.T)
    
    row_sums = np.sum(binary_matrix, axis=1, keepdims=True)
    
    union = row_sums + row_sums.T - intersection
    
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def jaccard_similarity_by_top_k(matrix, top_k=20, self_sim=False):
    binary_matrix = (matrix > 0).astype(int)
    
    intersection = np.dot(binary_matrix, binary_matrix.T)
    
    row_sums = np.sum(binary_matrix, axis=1, keepdims=True)
    
    union = row_sums + row_sums.T - intersection
    
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    for i in range(similarity_matrix.shape[0]):
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix

def cosine_similarity_by_threshold(matrix, threshold=0.0, self_sim=False):
    # Convert the matrix to binary (implicit feedback)
    binary_matrix = (matrix > 0).astype(int)
    
    # Compute cosine similarity on the binary matrix
    similarity_matrix = cosine_similarity(binary_matrix)
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def cosine_similarity_by_top_k(matrix, top_k=20, self_sim=False):
    binary_matrix = (matrix > 0).astype(int)
    
    similarity_matrix = cosine_similarity(binary_matrix)
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    for i in range(similarity_matrix.shape[0]):
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix

def tanimoto_similarity(matrix, threshold=0.0, self_sim=False):
    # Convert the matrix to binary (implicit feedback)
    binary_matrix = (matrix > 0).astype(int)
    
    # Calculate the dot product (intersection)
    dot_product = np.dot(binary_matrix, binary_matrix.T)
    
    # Calculate the sum of rows (for the union)
    sum_rows = np.sum(binary_matrix, axis=1, keepdims=True)
    
    # Compute the denominator (union)
    denominator = sum_rows + sum_rows.T - dot_product
    
    # Calculate Tanimoto (Jaccard) similarity
    similarity_matrix = dot_product / denominator
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix
    
def pearson_similarity_implicit(matrix, threshold=0.0, self_sim=False):
    # Convert matrix to boolean to handle implicit data (1/0)
    matrix = np.where(matrix > 0, 1, 0)
    
    # Calculate the sum of the matrix rows
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    
    # Handle cases where a row sum is zero (no interactions)
    row_sums[row_sums == 0] = 1  # Prevent division by zero

    # Normalize by the row sums (like a normalized interaction count)
    matrix_normalized = matrix / row_sums
    
    # Compute the dot product between all pairs of rows
    similarity_matrix = np.dot(matrix_normalized, matrix_normalized.T)
    
    # Compute the norms of the rows
    norms = np.linalg.norm(matrix_normalized, axis=1, keepdims=True)
    
    # Compute the outer product of norms
    norm_matrix = np.dot(norms, norms.T)
    
    # Avoid division by zero
    similarity_matrix = np.divide(similarity_matrix, norm_matrix, out=np.zeros_like(similarity_matrix), where=norm_matrix!=0)
    
    # Set diagonal to zero to remove self-loops
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def fusion_similarity_by_threshold(matrix, p_thresh=0.0, j_thresh=0.5, self_sim=False):
    # Compute Pearson similarity
    cos_matrix = cosine_similarity_by_threshold(matrix, threshold=p_thresh, self_sim=self_sim)

    # Compute Jaccard similarity
    jaccard_matrix = jaccard_similarity_by_threshold(matrix, threshold=j_thresh, self_sim=self_sim)

    # Combine Pearson and Jaccard similarities by multiplication
    combined_matrix = cos_matrix + jaccard_matrix

    return combined_matrix

def fusion_similarity_by_top_k(matrix, top_k=20, self_sim=False):
    # Compute Pearson similarity
    cos_matrix = cosine_similarity_by_top_k(matrix, top_k=top_k, self_sim=self_sim)

    # Compute Jaccard similarity
    jaccard_matrix = jaccard_similarity_by_top_k(matrix, top_k=top_k, self_sim=self_sim)

    # Combine Pearson and Jaccard similarities by multiplication
    combined_matrix = cos_matrix + jaccard_matrix

    return combined_matrix

def create_uuii_adjmat_by_threshold(df, u_sim='consine', i_sim='jaccard', u_sim_thresh=0.3, i_sim_thresh=0.3, self_sim=False):
    # Create user-item matrix
    
    ddf = dd.from_pandas(df, npartitions=10)
    user_item_matrix = ddf.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0).compute()

    #user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
    #user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_u_thresh)
    if u_sim == 'cosine':
        user_user_jaccard = cosine_similarity_by_threshold(user_item_matrix.values, threshold=u_sim_thresh, self_sim=self_sim)
    elif u_sim == 'tanimoto':
        user_user_jaccard = tanimoto_similarity(user_item_matrix.values, threshold=u_sim_thresh, self_sim=self_sim)
    else:
        user_user_jaccard = jaccard_similarity_by_threshold(user_item_matrix.values, threshold=u_sim_thresh, self_sim=self_sim)
    
    if i_sim == 'cosine':
        item_item_jaccard = cosine_similarity_by_threshold(user_item_matrix.T.values, threshold=i_sim_thresh, self_sim=self_sim)
    elif i_sim == 'tanimoto':
        item_item_jaccard = tanimoto_similarity(user_item_matrix.T.values, threshold=i_sim_thresh, self_sim=self_sim)
    else:
        item_item_jaccard = jaccard_similarity_by_threshold(user_item_matrix.T.values, threshold=i_sim_thresh, self_sim=self_sim)
    
    user_user_adjacency = user_user_jaccard
    item_item_adjacency = item_item_jaccard
    
    # Dimensions
    num_users = user_user_adjacency.shape[0]
    num_items = item_item_adjacency.shape[0]
    total_size = num_users + num_items

    # Initialize combined adjacency matrix
    combined_adjacency = np.zeros((total_size, total_size))

    # Fill in the user-user and item-item parts
    combined_adjacency[:num_users, :num_users] = user_user_adjacency
    combined_adjacency[num_users:, num_users:] = item_item_adjacency
    
    # Convert to DataFrame for readability
    combined_adjacency_df = pd.DataFrame(combined_adjacency)
    
    return combined_adjacency_df


def create_uuii_adjmat_by_top_k(df, u_sim='consine', i_sim='jaccard', u_sim_top_k=20, i_sim_top_k=20, self_sim=False):
    # Create user-item matrix
    #ddf = dd.from_pandas(df, npartitions=10)
    #user_item_matrix = ddf.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0).compute()
    #user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
    # Convert to NumPy arrays
    user_ids = df['user_id_idx'].to_numpy()
    item_ids = df['item_id_idx'].to_numpy()

    # Create a sparse matrix directly
    user_item_matrix_coo = coo_matrix((np.ones(len(df)), (user_ids, item_ids)))
    user_item_matrix = user_item_matrix_coo.toarray()

    #user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_u_thresh)
    if u_sim == 'cosine':
        user_user_sim_matrix = cosine_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim)
    elif u_sim == 'mix':
        user_user_sim_matrix = fusion_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim)
    else:
        user_user_sim_matrix = jaccard_similarity_by_top_k(user_item_matrix, top_k=u_sim_top_k, self_sim=self_sim)
    
    if i_sim == 'cosine':
        item_item_sim_matrix = cosine_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim)
    elif i_sim == 'mix':
        item_item_sim_matrix = fusion_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim)
    else:
        item_item_sim_matrix = jaccard_similarity_by_top_k(user_item_matrix.T, top_k=i_sim_top_k, self_sim=self_sim)
    
    user_user_adjacency = user_user_sim_matrix
    item_item_adjacency = item_item_sim_matrix
    
    # Dimensions
    num_users = user_user_adjacency.shape[0]
    num_items = item_item_adjacency.shape[0]
    total_size = num_users + num_items

    # Initialize combined adjacency matrix
    combined_adjacency = np.zeros((total_size, total_size))

    # Fill in the user-user and item-item parts
    combined_adjacency[:num_users, :num_users] = user_user_adjacency
    combined_adjacency[num_users:, num_users:] = item_item_adjacency
    
    # Convert to DataFrame for readability
    combined_adjacency_df = pd.DataFrame(combined_adjacency)
    
    return combined_adjacency_df


def load_data(dataset = "ml-100k", min_interaction_threshold = 20, verbose = 0):
    
    user_df = None
    item_df = None
    ratings_df = None
    rating_stat = None
    
    if dataset == 'ml-latest-small':
        movies_path = f'../data/{dataset}/movies.csv'
        ratings_path = f'../data/{dataset}/ratings.csv'

        # Load the entire ratings dataframe into memory:
        ratings_df = pd.read_csv(ratings_path)[["userId", "movieId", "rating", "timestamp"]]
        
        ratings_df = ratings_df.rename(columns={'movieId': 'itemId'})

        # Load the entire movie dataframe into memory:
        item_df = pd.read_csv(movies_path)
        item_df = item_df.rename(columns={'movieId': 'itemId'})
        item_df = item_df.set_index('itemId')
        
    elif dataset == 'ml-100k':
        # Paths for ML-100k data files
        ratings_path = f'data/{dataset}/u.data'
        movies_path = f'data/{dataset}/u.item'
        users_path = f'data/{dataset}/u.user'
        
        # Load the entire ratings dataframe into memory
        ratings_df = pd.read_csv(ratings_path, sep='\t', names=["userId", "itemId", "rating", "timestamp"])

        # Load the entire movie dataframe into memory
        genre_columns = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        item_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=["itemId", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_columns)
        
        # Create the genres column by concatenating genre names where the value is 1
        item_df['genres'] = item_df[genre_columns].apply(lambda row: '|'.join([genre for genre, val in row.items() if val == 1]), axis=1)
    
        # Keep only the necessary columns
        item_df = item_df[["itemId", "title", "genres"]]
        item_df = item_df.set_index('itemId')
        
        # Load the entire user dataframe into memory 1|24|M|technician|85711
        user_df = pd.read_csv(users_path, sep='|', encoding='latin-1', names=["userId", "age_group", "sex", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('userId')
    
    elif dataset == 'ml-1m':
        # Paths for ML-1M data files
        ratings_path = f'data/{dataset}/ratings.dat'
        movies_path = f'data/{dataset}/movies.dat'
        users_path = f'data/{dataset}/users.dat'

        # Load the entire ratings dataframe into memory
        ratings_df = pd.read_csv(ratings_path, sep='::', names=["userId", "itemId", "rating", "timestamp"], engine='python', encoding='latin-1')
        
        # Load the entire movie dataframe into memory
        item_df = pd.read_csv(movies_path, sep='::', names=["itemId", "title", "genres"], engine='python', encoding='latin-1')
        item_df = item_df.set_index('itemId')
        
        # Load the entire user dataframe into memory UserID::Gender::Age::Occupation::Zip-code -> 1::F::1::10::48067 
        user_df = pd.read_csv(users_path, sep='::', encoding='latin-1', names=["userId", "sex", "age_group", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('userId')
        
    elif dataset == 'amazon_cloth':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/df_modcloth.csv'

        # Load the entire ratings dataframe into memory        
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['item_id', 'user_id', 'rating', 'timestamp']]
        # Rename the columns
        df_selected.columns = ['itemId', 'userId', 'rating', 'timestamp']
        
        # Parse timestamps and remove timezone information if present        
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp'], errors='coerce')
        # Convert to Unix time in seconds
        df_selected['timestamp'] = df_selected['timestamp'].astype(int) // 10**9

        # Option 2: Drop rows with NA timestamps
        #df_selected = df_selected.dropna(subset=['timestamp'])
        
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df_selected['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index

        df_filtered = df_selected[df_selected['userId'].isin(filtered_users)]
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
    
    elif dataset == 'amazon_fashion':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/amazon_fashion.csv'
        
        # Load the entire ratings dataframe into memory
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]

        # Rename the columns
        df_selected.columns = ['userId', 'itemId', 'rating', 'timestamp']
                
        # Option 2: Drop rows with NA timestamps
        # df_selected = df_selected.dropna(subset=['timestamp'])
      
         # Filter users with at least a minimum number of interactions
        user_interaction_counts = df_selected['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df_selected[df_selected['userId'].isin(filtered_users)]
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    elif dataset == 'epinion':
        # Paths for ML-1M data files
        ratings_path = f'data/epinion/rating_with_timestamp.txt'
        
        # Read the text file into a DataFrame
        df = pd.read_csv('data/epinion/rating_with_timestamp.txt', sep=r'\s+', header=None)

        # Assign column names
        df.columns = ['userId', 'itemId', 'categoryId', 'rating', 'helpfulness', 'timestamp']

        # Select only the columns we need
        df_selected = df[['userId', 'itemId', 'rating', 'timestamp']]

        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df_selected['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df_selected[df_selected['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
    
    elif dataset == 'douban_book':
        ratings_path = f'data/douban/bookreviews_cleaned.txt'
            
        # Read the text file into a DataFrame
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'book_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'book_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    elif dataset == 'douban_music':
        # Read the text file into a DataFrame
        ratings_path = f'data/douban/musicreviews_cleaned.txt'
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'music_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'music_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    elif dataset == 'douban_movie':
        # Read the text file into a DataFrame
        ratings_path = f'data/douban/moviereviews_cleaned.txt'
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'movie_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'movie_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
    
    elif dataset == 'yelp':
        # Read the text file into a DataFrame
        ratings_path = f'data/yelp/yelp_reviews.csv'
        df = pd.read_csv(ratings_path, sep=',')

        # Select and rename the columns to match the desired format
        df = df[['user_id', 'item_id', 'rating', 'timestamp']]
        df = df.rename(columns={'user_id': 'userId', 'item_id': 'itemId'})
                
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        print('Number of users before threshold {min_interaction_threshold}:', len(df['userId'].unique()))
        # Filter users with at least a minimum number of interactions
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interaction_threshold].index
        
        print('Number of users after threshold {min_interaction_threshold}:', len(filtered_users))
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
    

    if ratings_df is not None:
        
        ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'itemId': 'item_id'})
        #_lbl_user = preprocessing.LabelEncoder()
        #_lbl_movie = preprocessing.LabelEncoder()
        
        #ratings_df.userId = _lbl_user.fit_transform(ratings_df.userId.values)
        #ratings_df.itemId = _lbl_movie.fit_transform(ratings_df.itemId.values)
            
        num_users = len(ratings_df['user_id'].unique())
        num_items = len(ratings_df['item_id'].unique())
        mean_rating = round(ratings_df['rating'].mean(), 2)
        num_ratings = len(ratings_df)
        
        # Calculate the max-min time distance
        min_timestamp = ratings_df['timestamp'].min()
        max_timestamp = ratings_df['timestamp'].max()
        max_min_time_distance = round((max_timestamp - min_timestamp) / 86400, 0)
        
        rating_stat = {'num_users': num_users, 'num_items': num_items, 'mean_rating': mean_rating, 'num_ratings': num_ratings, 'time_distance': max_min_time_distance}

        if verbose > -1:
            print(f'{br}{dataset}{rs} | {rating_stat}')
        elif verbose == 1:
            print(ratings_df.head())
        
    else:
        print(f'{br}No data is loaded for dataset: {dataset} !!! {rs}')
        
    return ratings_df, user_df, item_df, rating_stat