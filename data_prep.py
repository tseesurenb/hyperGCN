'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

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
    combined_matrix = cos_matrix * jaccard_matrix

    return combined_matrix

def fusion_similarity_by_top_k(matrix, top_k=20, self_sim=False):
    # Compute Pearson similarity
    cos_matrix = cosine_similarity_by_top_k(matrix, top_k=top_k, self_sim=self_sim)

    # Compute Jaccard similarity
    jaccard_matrix = jaccard_similarity_by_top_k(matrix, top_k=top_k, self_sim=self_sim)

    # Combine Pearson and Jaccard similarities by multiplication
    combined_matrix = cos_matrix * jaccard_matrix

    return combined_matrix

def create_uuii_adjmat_by_threshold(df, u_sim='consine', i_sim='jaccard', u_sim_thresh=0.3, i_sim_thresh=0.3, self_sim=False):
    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
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
    user_item_matrix = df.pivot_table(index='user_id_idx', columns='item_id_idx', values='rating', fill_value=0)
    
    #user_user_jaccard = jaccard_similarity(user_item_matrix.values, threshold=j_u_thresh)
    if u_sim == 'cosine':
        user_user_sim_matrix = cosine_similarity_by_top_k(user_item_matrix.values, top_k=u_sim_top_k, self_sim=self_sim)
    elif u_sim == 'mix':
        user_user_sim_matrix = fusion_similarity_by_top_k(user_item_matrix.values, top_k=u_sim_top_k, self_sim=self_sim)
    else:
        user_user_sim_matrix = jaccard_similarity_by_top_k(user_item_matrix.values, top_k=u_sim_top_k, self_sim=self_sim)
    
    if i_sim == 'cosine':
        item_item_sim_matrix = cosine_similarity_by_top_k(user_item_matrix.T.values, top_k=i_sim_top_k, self_sim=self_sim)
    elif i_sim == 'mix':
        item_item_sim_matrix = fusion_similarity_by_top_k(user_item_matrix.T.values, top_k=i_sim_top_k, self_sim=self_sim)
    else:
        item_item_sim_matrix = jaccard_similarity_by_top_k(user_item_matrix.T.values, top_k=i_sim_top_k, self_sim=self_sim)
    
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
