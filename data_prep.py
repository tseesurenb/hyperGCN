# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def create_uuii_adjmat(df):
    # Unique users and items
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()

    # Create mappings
    user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
    item_mapping = {item_id: i + len(unique_users) for i, item_id in enumerate(unique_items)}

    # Apply mappings
    df['user_id_idx'] = df['user_id'].map(user_mapping)
    df['item_id_idx'] = df['item_id'].map(item_mapping)

    # Create user-item matrix
    user_item_matrix = df.pivot(index='user_id_idx', columns='item_id_idx', values='item_id').notnull().astype(int)

    # Compute user-user adjacency matrix
    user_user_adjacency = user_item_matrix.dot(user_item_matrix.T)
    user_user_adjacency[user_user_adjacency > 0] = 1  # Convert to binary adjacency matrix

    # Compute item-item adjacency matrix
    item_user_matrix = user_item_matrix.T
    item_item_adjacency = item_user_matrix.dot(item_user_matrix.T)
    item_item_adjacency[item_item_adjacency > 0] = 1  # Convert to binary adjacency matrix

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


def create_uuii_adjmat2(df):
       
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='item_id', aggfunc='count', fill_value=0)

    # Compute user-user adjacency matrix (number of common items)
    user_user_adjacency = user_item_matrix.dot(user_item_matrix.T)
    np.fill_diagonal(user_user_adjacency.values, 0)  # Remove self-loops

    # Compute item-item adjacency matrix (number of common users)
    item_user_matrix = user_item_matrix.T
    item_item_adjacency = item_user_matrix.dot(item_user_matrix.T)
    np.fill_diagonal(item_item_adjacency.values, 0)  # Remove self-loops
    
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

def jaccard_similarity_by_threshold(matrix, threshold=0.0):
    
    binary_matrix = (matrix > 0).astype(int)
    
    intersection = np.dot(binary_matrix, binary_matrix.T)
    
    row_sums = np.sum(binary_matrix, axis=1, keepdims=True)
    
    union = row_sums + row_sums.T - intersection
    
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    np.fill_diagonal(similarity_matrix, 0)
    
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def jaccard_similarity_by_top_k(matrix, top_k=20):
    binary_matrix = (matrix > 0).astype(int)
    
    intersection = np.dot(binary_matrix, binary_matrix.T)
    
    row_sums = np.sum(binary_matrix, axis=1, keepdims=True)
    
    union = row_sums + row_sums.T - intersection
    
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    np.fill_diagonal(similarity_matrix, 0)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    for i in range(similarity_matrix.shape[0]):
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix

def cosine_similarity_by_threshold(matrix, threshold=0.0):
    # Convert the matrix to binary (implicit feedback)
    binary_matrix = (matrix > 0).astype(int)
    
    # Compute cosine similarity on the binary matrix
    similarity_matrix = cosine_similarity(binary_matrix)
    
    # Set the diagonal to zero (no self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def cosine_similarity_by_top_k(matrix, top_k=20):
    binary_matrix = (matrix > 0).astype(int)
    
    similarity_matrix = cosine_similarity(binary_matrix)
    
    np.fill_diagonal(similarity_matrix, 0)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    for i in range(similarity_matrix.shape[0]):
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix

def tanimoto_similarity(matrix, threshold=0.0):
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
    
    # Set the diagonal to zero (no self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix
    
def pearson_similarity(matrix, threshold=0.0):
    # Normalize the matrix (subtract mean of each row)
    mean_ratings = np.mean(matrix, axis=1, keepdims=True)
    matrix_normalized = matrix - mean_ratings
    
    print(matrix_normalized)
    
    # Compute the dot product between all pairs of rows
    similarity_matrix = np.dot(matrix_normalized, matrix_normalized.T)
    
    # Compute the norms of the rows
    norms = np.linalg.norm(matrix_normalized, axis=1, keepdims=True)
    
    # Compute the outer product of norms
    norm_matrix = np.dot(norms, norms.T)
    
    # Avoid division by zero
    similarity_matrix = np.divide(similarity_matrix, norm_matrix, out=np.zeros_like(similarity_matrix), where=norm_matrix!=0)
    
    # Set diagonal to zero to remove self-loops
    np.fill_diagonal(similarity_matrix, 0)
    
    # Apply the threshold
    similarity_matrix[similarity_matrix < threshold] = 0
    
    return similarity_matrix

def combined_similarity(matrix, p_thresh=0.0, j_thresh=0.5):
    # Compute Pearson similarity
    mean_ratings = np.mean(matrix, axis=1, keepdims=True)
    matrix_normalized = matrix - mean_ratings
    pearson_matrix = np.dot(matrix_normalized, matrix_normalized.T)
    norms = np.linalg.norm(matrix_normalized, axis=1, keepdims=True)
    norm_matrix = np.dot(norms, norms.T)
    pearson_matrix = np.divide(pearson_matrix, norm_matrix, out=np.zeros_like(pearson_matrix), where=norm_matrix != 0)
    np.fill_diagonal(pearson_matrix, 0)
    pearson_matrix[pearson_matrix < p_thresh] = 0
    pearson_matrix = (pearson_matrix + 1) / 2  # Scale Pearson similarity to [0, 1]

    # Compute Jaccard similarity
    binary_matrix = (matrix > 0).astype(int)
    intersection = np.dot(binary_matrix, binary_matrix.T)
    row_sums = np.sum(binary_matrix, axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    jaccard_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    np.fill_diagonal(jaccard_matrix, 0)
    jaccard_matrix[jaccard_matrix < j_thresh] = 0

    # Combine Pearson and Jaccard similarities by multiplication
    combined_matrix = pearson_matrix * jaccard_matrix

    return combined_matrix
    
def create_pearson_sim_uuii_adjmat(df, p_thresh=0.0, j_thresh=0.5):
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)

    p_thresh = p_thresh
    # Compute user-user Pearson similarity matrix
    user_user_pearson = pearson_similarity(user_item_matrix.values, threshold=p_thresh)
    # print(user_user_adjacency)

    # Compute item-item Pearson similarity matrix
    item_user_matrix = user_item_matrix.T
    item_item_pearson = pearson_similarity(item_user_matrix.values, threshold=p_thresh)
    
    j_thresh = j_thresh
    
    user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_thresh)
    item_item_jaccard = jaccard_similarity(user_item_matrix.T.values, threshold=j_thresh)
    
    user_user_adjacency = user_user_pearson * user_user_jaccard
    item_item_adjacency = item_item_pearson * item_item_jaccard
    
    #user_user_adjacency = user_user_jaccard
    #item_item_adjacency = item_item_jaccard
    
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

def create_jaccard_uuii_adjmat(df, u_sim='consine', i_sim='jaccard', u_sim_thresh=0.3, i_sim_thresh=0.3):
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
    #user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_u_thresh)
    if u_sim == 'cosine':
        user_user_jaccard = cosine_similarity_by_threshold(user_item_matrix.values, threshold=u_sim_thresh)
    elif u_sim == 'tanimoto':
        user_user_jaccard = tanimoto_similarity(user_item_matrix.values, threshold=u_sim_thresh)
    else:
        user_user_jaccard = jaccard_similarity_by_threshold(user_item_matrix.values, threshold=u_sim_thresh)
    
    if i_sim == 'cosine':
        item_item_jaccard = cosine_similarity_by_threshold(user_item_matrix.T.values, threshold=i_sim_thresh)
    elif i_sim == 'tanimoto':
        item_item_jaccard = tanimoto_similarity(user_item_matrix.T.values, threshold=i_sim_thresh)
    else:
        item_item_jaccard = jaccard_similarity_by_threshold(user_item_matrix.T.values, threshold=i_sim_thresh)
    
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


def create_uuii_adjmat_top_k(df, u_sim='consine', i_sim='jaccard', u_sim_top_k=20, i_sim_top_k=20):
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
    #user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_u_thresh)
    if u_sim == 'cosine':
        user_user_jaccard = cosine_similarity_by_top_k(user_item_matrix.values, top_k=u_sim_top_k)
    else:
        user_user_jaccard = jaccard_similarity_by_top_k(user_item_matrix.values, top_k=u_sim_top_k)
    
    if i_sim == 'cosine':
        item_item_jaccard = cosine_similarity_by_top_k(user_item_matrix.T.values, top_k=i_sim_top_k)
    else:
        item_item_jaccard = jaccard_similarity_by_top_k(user_item_matrix.T.values, top_k=i_sim_top_k)
    
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

def create_jaccard_uuii_adjmat_coo(df, j_u_thresh=0.5, j_i_thresh=0.5):
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)

    # Calculate Jaccard similarity matrices
    user_user_jaccard = jaccard_similarity_by_threshold(user_item_matrix.values, threshold=j_u_thresh)
    item_item_jaccard = jaccard_similarity_by_threshold(user_item_matrix.T.values, threshold=j_i_thresh)
    
    # Dimensions
    num_users = user_user_jaccard.shape[0]
    num_items = item_item_jaccard.shape[0]
    total_size = num_users + num_items

    # Initialize row, col, and data lists for the COO matrix
    rows, cols, data = [], [], []

    # Fill in the user-user part
    user_user_indices = np.nonzero(user_user_jaccard)
    rows.extend(user_user_indices[0])
    cols.extend(user_user_indices[1])
    data.extend(user_user_jaccard[user_user_indices])

    # Fill in the item-item part
    item_item_indices = np.nonzero(item_item_jaccard)
    rows.extend(item_item_indices[0] + num_users)  # Shift item indices by num_users
    cols.extend(item_item_indices[1] + num_users)
    data.extend(item_item_jaccard[item_item_indices])

    # Create sparse COO matrix
    combined_adjacency_coo = coo_matrix((data, (rows, cols)), shape=(total_size, total_size))

    return combined_adjacency_coo