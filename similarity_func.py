import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix


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


def cosine_similarity_by_top_k_new(matrix, top_k=20, self_sim=False, verbose=-1):
    
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    # Convert the binary matrix to a sparse matrix (CSR format)
    binary_matrix = (matrix > 0).astype(int)
    sparse_matrix = csr_matrix(binary_matrix)

    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # If self_sim is False, set the diagonal to zero
    if not self_sim:
        similarity_matrix.setdiag(0)
    else:
        similarity_matrix.setdiag(1)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col

        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    return filtered_similarity_matrix.tocsr()

def cosine_similarity_by_top_k_old(matrix, top_k=20, self_sim=False, verbose=-1):
    
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    binary_matrix = (matrix > 0).astype(int)
    
    similarity_matrix = cosine_similarity(binary_matrix)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix


def cosine_similarity_by_top_k(matrix, top_k=20, self_sim=False, verbose=-1):
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    # Convert to binary matrix only if necessary
    binary_matrix = (matrix > 0).astype(int) if not np.issubdtype(matrix.dtype, np.bool_) else matrix

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(binary_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # Set self-similarity values
    if not self_sim:
        similarity_matrix.setdiag(0)
    
    # Efficient top-K filtering
    if verbose > 0:
        print('Filtering top-k values...')
    
    top_k_indices = np.argpartition(-similarity_matrix.data, top_k - 1, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix.toarray())

    pbar = tqdm(range(similarity_matrix.shape[0]), 
                bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', 
                ascii="░❯", disable=(verbose <= 0))
    pbar.set_description(f'Preparing similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        rows, cols = similarity_matrix[i].nonzero()
        filtered_similarity_matrix[i, cols[top_k_indices[i]]] = similarity_matrix[i, cols[top_k_indices[i]]]
    
    return filtered_similarity_matrix