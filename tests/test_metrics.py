import torch
import pandas as pd
import numpy as np

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_data, K, device):
    test_user_ids = torch.LongTensor(test_data['user_id'].unique()).to(device)
    
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)
    
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
    print("Relevance Score:\n", relevance_score)

    # create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    v = torch.ones((len(train_df)), dtype=torch.float32).to(device)
    
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
    print("Interactions Tensor:\n", interactions_t)

    # mask out training user-item interactions from metric computation
    relevance_score = relevance_score * (1 - interactions_t)
    print("Masked Relevance Score:\n", relevance_score)

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    print("Top K Relevance Indices:\n", topk_relevance_indices)
    
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID', 'top_rlvnt_itm']]
    print("Top K Relevance DataFrame:\n", topk_relevance_indices_df)

    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id')['item_id'].apply(list).reset_index()
    print("Metrics DataFrame with Test Interactions:\n", test_interacted_items)

    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id', right_on='user_ID')
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]
    print("Metrics DataFrame with Intersection Items:\n", metrics_df)

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    # Calculate nDCG
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(relevance_scores, k):
        dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(relevance_scores, k) / dcg_max

    metrics_df['ndcg'] = metrics_df.apply(lambda x: ndcg_at_k([1 if i in x['item_id'] else 0 for i in x['top_rlvnt_itm']], K), axis=1)
    
    # Print final metrics dataframe
    print("Final Metrics DataFrame:\n", metrics_df)

    # Return mean recall, precision, and nDCG
    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean()

# Example call (test this with actual data)
user_Embed_wts = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
item_Embed_wts = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9], [1.0, 1.1]])
n_users = 4
n_items = 5
train_df = pd.DataFrame({'user_id': [0, 1, 2, 3], 'item_id': [0, 1, 2, 3]})
test_data = pd.DataFrame({'user_id': [0, 1, 2, 3], 'item_id': [1, 2, 3, 4]})
K = 2
device = torch.device('cpu')

recall, precision, ndcg = get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_data, K, device)
print(f"Recall: {recall}, Precision: {precision}, nDCG: {ndcg}")

