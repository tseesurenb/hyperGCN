import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

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
    print('Computing cosine similarity by top-k...')
    
    # Convert the binary matrix to a sparse matrix (CSR format)
    binary_matrix = (matrix > 0).astype(int)
    sparse_matrix = csr_matrix(binary_matrix)

    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
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

def cosine_similarity_by_top_k_old(matrix, top_k=20, self_sim=False):
    print('Computing cosine similarity by top-k...')
    binary_matrix = (matrix > 0).astype(int)
    
    # Convert the binary matrix to a sparse matrix
    sparse_matrix = csr_matrix(binary_matrix)

    # Compute sparse cosine similarity
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    #similarity_matrix = cosine_similarity(binary_matrix)
    
    print('Cosine similarity computed.')
    
    similarity_matrix = similarity_matrix.toarray() 
    
    if not self_sim:
        np.fill_diagonal(similarity_matrix, 0) # Set the diagonal to zero (no self-similarity)
    else:
        np.fill_diagonal(similarity_matrix, 1)
    
    # Filter top K values for each row
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    
    print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        filtered_similarity_matrix[i, top_k_indices[i]] = similarity_matrix[i, top_k_indices[i]]
    
    return filtered_similarity_matrix

from scipy.sparse import csr_matrix

def cosine_similarity_by_top_k_new(matrix, top_k=20, self_sim=False):
    print('Computing cosine similarity by top-k...')
    # Convert to sparse matrix
    binary_matrix = csr_matrix((matrix > 0).astype(int))
    
    # Initialize an empty list to store top_k similarities
    data, rows, cols = [], [], []
    
    # Iterate through each row to compute top_k similarities
    for i in tqdm(range(binary_matrix.shape[0])):
        # Compute the cosine similarity for the i-th row
        similarity_vector = cosine_similarity(binary_matrix[i], binary_matrix).flatten()
        
        if not self_sim:
            similarity_vector[i] = 0  # Set self-similarity to zero
        else:
            similarity_vector[i] = 1  # Set self-similarity to one
        
        # Get the indices of the top_k similarities
        top_k_indices = np.argsort(-similarity_vector)[:top_k]
        
        # Store the top_k values
        data.extend(similarity_vector[top_k_indices])
        rows.extend([i] * top_k)
        cols.extend(top_k_indices)
    
    # Construct the sparse similarity matrix using the top_k values
    filtered_similarity_matrix = csr_matrix((data, (rows, cols)), shape=(matrix.shape[0], matrix.shape[0]))

    print('Cosine similarity by top-k computed.')
    
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