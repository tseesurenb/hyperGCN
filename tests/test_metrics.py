import torch
import pandas as pd
import numpy as np

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_metrics

def test_get_metrics():
    # Create dummy data
    n_users = 3
    n_items = 3

    user_Embed_wts = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    item_Embed_wts = torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]])

    train_data = pd.DataFrame({
        'user_id': [0, 1, 2],
        'item_id': [0, 1, 2]
    })

    test_data = pd.DataFrame({
        'user_id': [0, 1, 2],
        'item_id': [1, 2, 0]
    })

    K = 2
    device = torch.device('cpu')  # Change to 'cuda' if you are using GPU

    # Run the function
    recall, precision, ndcg = get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K, device)

    # Expected values for the dummy data
    expected_recall = 1.0  # All items in test data are recommended in the top K
    expected_precision = 1.0  # Precision is 1 as we have perfect recommendations
    expected_ndcg = 1.0  # Perfect nDCG score

    # Print results
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"nDCG: {ndcg}")

    # Check if results are as expected
    assert np.isclose(recall, expected_recall), f"Expected recall: {expected_recall}, but got: {recall}"
    assert np.isclose(precision, expected_precision), f"Expected precision: {expected_precision}, but got: {precision}"
    assert np.isclose(ndcg, expected_ndcg), f"Expected nDCG: {expected_ndcg}, but got: {ndcg}"

# Run the test
test_get_metrics()



 # Step 1: Make sure test data has only user-item pairs that are in the training data
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()
    
    test_df = test_df[
      (test_df['user_id'].isin(train_user_ids)) & \
      (test_df['item_id'].isin(train_item_ids))
    ]
    
    # Step 2: Encode user and item IDs
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train_df['user_id'] = le_user.fit_transform(train_df['user_id'].values)
    train_df['item_id'] = le_item.fit_transform(train_df['item_id'].values)

    test_df['user_id'] = le_user.transform(test_df['user_id'].values)
    test_df['item_id'] = le_item.transform(test_df['item_id'].values)
