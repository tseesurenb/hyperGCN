import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack, hstack


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


import torch

# Assuming your sim functions are defined or imported elsewhere
# Mock similarity functions for testing
class sim:
    @staticmethod
    def cosine_similarity_by_top_k(matrix, top_k, self_sim, verbose):
        return coo_matrix(np.random.rand(matrix.shape[0], matrix.shape[0]))

    @staticmethod
    def jaccard_similarity_by_top_k(matrix, top_k, self_sim, verbose):
        return coo_matrix(np.random.rand(matrix.shape[0], matrix.shape[0]))

# Mock DataFrame with 5 users and 4 items
df = pd.DataFrame({
    'user_id': [0, 0, 1, 2, 3, 4],
    'item_id': [0, 1, 2, 0, 1, 3]
})

# Test case: Check get_edge_index output with 5 users and 4 items
def test_get_edge_index():
    # Generate adjacency matrix using the function create_uuii_adjmat
    adj_matrix = create_uuii_adjmat(df, u_sim='cosine', i_sim='cosine', u_sim_top_k=5, i_sim_top_k=5, self_sim=False, verbose=1)
    
    # Get edge index and data from the sparse matrix using get_edge_index
    edge_index, data = get_edge_index(adj_matrix)
    
    # Check the edge index and data dimensions
    print("Edge Index:")
    print(edge_index)
    print("Data Values:")
    print(data)

    # Check if indices are within the expected range (0 to 8)
    assert edge_index.min() >= 0, "Edge index contains negative values!"
    assert edge_index.max() < (5 + 4), "Edge index exceeds user and item range!"
    
    # Optional: Check data length consistency with edge index size
    assert edge_index.shape[1] == len(data), "Edge index and data length mismatch!"
    print("Test passed!")

# Run the test case
test_get_edge_index()
