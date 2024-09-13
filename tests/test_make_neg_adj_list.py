import pandas as pd

# Example Data
data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3],
    'item_id': [101, 102, 103, 104, 101]
})

# All possible items
all_items = [101, 102, 103, 104, 105]

# The function implementation
def make_neg_adj_list(data, all_items):
    all_items_set = set(all_items)

    # Group by user_id and aggregate item_ids into lists
    pos_items = data.groupby('user_id')['item_id'].agg(list)
    
    # Compute neg_items by subtracting the pos_items from all_items for each user
    neg_items = pos_items.apply(lambda pos: list(all_items_set.difference(pos)))
    
    # Create a dictionary with user_id as the key and neg_items as the value
    neg_adj_list_dict = pd.Series(neg_items, index=pos_items.index).to_dict()
    
    return neg_adj_list_dict

# Expected output
expected_output = {
    1: [105, 103, 104],  # Negative items for user 1 (remaining items after excluding 101, 102)
    2: [101, 102, 105],  # Negative items for user 2 (remaining items after excluding 103, 104)
    3: [102, 103, 104, 105]  # Negative items for user 3 (remaining items after excluding 101)
}

# Run the function
neg_adj_list = make_neg_adj_list(data, all_items)

# Convert lists to sets to compare unordered elements
neg_adj_list_sets = {user_id: set(items) for user_id, items in neg_adj_list.items()}
expected_output_sets = {user_id: set(items) for user_id, items in expected_output.items()}

# Output result and compare with expected output
print("Generated Negative Adjacency List:", neg_adj_list_sets)
assert neg_adj_list_sets == expected_output_sets, f"Test failed: {neg_adj_list_sets} != {expected_output_sets}"

# If no assertion error, print success message
print("Test passed: Output matches the expected result.")
