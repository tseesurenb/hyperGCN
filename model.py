import torch
from torch import nn, Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import degree, softmax as geo_softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GCNConv(MessagePassing):
    def __init__(self, num_users, num_items, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = 3
        self.in_channels = in_channels 
        self.out_channels = out_channels
        
        self.num_nodes = num_users + num_items
        
        self.f = nn.ReLU()
        
        self.ui_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.in_channels)
        self.ui_emb.weight.requires_grad = True
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, label_indices, r_mat_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        #u_emb_0 = self.users_emb.weight + self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight
        #u_emb_0 = self.users_emb.weight
        #i_emb_0 = self.items_emb.weight
        
        ui_emb_0 = self.ui_emb.weight
        
        #u_emb_0 = self.users_emb.weight + self._u_base_emb.weight
        #i_emb_0 = self.items_emb.weight + self._i_base_emb.weight
        
        emb_0 = ui_emb_0
        embs = [emb_0]
        emb_k = emb_0
        
        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index # this is in COO format
        deg = degree(col, self.num_nodes, dtype=emb_k.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        #out = self.propagate(edge_index, x=x, norm=norm)
        
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=edge_index, x=emb_k, norm=norm)
            embs.append(emb_k)
            
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        #users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # Step 6: convert COO format index to Interactions format index
        #src = r_mat_index[0]
        #dest = r_mat_index[1]
        #user_embeds = users_emb_final[src]
        #item_embeds = items_emb_final[dest]
        
        #user_embeds = users_emb_final[row]
        #item_embeds = items_emb_final[col]
        
        src, dest = label_indices
        
        user_embeds = emb_final[src]
        item_embeds = emb_final[dest]
        
        
        out = torch.mul(user_embeds, item_embeds)

        # Step 6: Apply a final bias vector.
        out = out + self.bias
        
        out = torch.sum(out, dim=-1)
        
        out = self.f(out)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class LightAttnGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        #out_features = 64
        #in_features = 64
        #self.att_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        #self.a_att_weight = torch.nn.Parameter(torch.Tensor(1))
        #self.r_att_weight = torch.nn.Parameter(torch.Tensor(1))
        
        self.in_features = 1
        self.out_features = 1
        self.num_heads = 3

        self.a_att_weights = torch.nn.Parameter(torch.Tensor(self.num_heads, self.in_features))
        self.r_att_weights = torch.nn.Parameter(torch.Tensor(self.num_heads, self.in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_uniform_(self.att_weight)
        #torch.nn.init.xavier_uniform_(self.a_att_weight.view(1, -1))
        #torch.nn.init.xavier_uniform_(self.r_att_weight.view(1, -1))
        
        torch.nn.init.xavier_uniform_(self.a_att_weights)
        torch.nn.init.xavier_uniform_(self.r_att_weights)
        
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
         
        a_edge_attr = edge_attrs[:, 0].unsqueeze(1)
        r_edge_attr = edge_attrs[:, 1].unsqueeze(1)
             
        # Compute attention weights from edge attributes using multi-head attention
        a_attention_scores = [F.leaky_relu(a_edge_attr * self.a_att_weights[i]) for i in range(self.num_heads)]
        r_attention_scores = [F.leaky_relu(r_edge_attr * self.r_att_weights[i]) for i in range(self.num_heads)]

        # Sum the attention scores from all heads
        a_attention_weights = torch.softmax(sum(a_attention_scores), dim=0)
        r_attention_weights = torch.softmax(sum(r_attention_scores), dim=0)

        # Integrate attention weights into normalization factor
        norm = norm * a_attention_weights.view(-1) * r_attention_weights.view(-1)
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    
class TempLGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        #out_features = 64
        #in_features = 64
        #self.att_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.a_att_weight = torch.nn.Parameter(torch.Tensor(1))
        self.r_att_weight = torch.nn.Parameter(torch.Tensor(1))
             
        self.reset_parameters()

    def reset_parameters(self):        
        torch.nn.init.xavier_uniform_(self.a_att_weight)
        torch.nn.init.xavier_uniform_(self.r_att_weight)
        
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        #norm = gcn_norm(edge_index=edge_index, add_self_loops=False)
        
        # Compute attention weights from edge attributes (timestamps)
        #edge_attr = edge_attr.view(-1, 1)
        #attention_scores = F.leaky_relu(edge_attr * self.att_weight)
        #attention_weights = torch.softmax(attention_scores, dim=0)

        # Integrate attention weights into normalization factor
        #norm = norm * attention_weights.view(-1)
        
        a_edge_attr = edge_attrs[:, 0].unsqueeze(1)
        r_edge_attr = edge_attrs[:, 1].unsqueeze(1)
     
        # Compute attention weights from edge attributes (timestamps)
        #edge_attr = edge_attr.view(-1, 1)
        #attention_scores = F.leaky_relu(torch.matmul(edge_attr, self.att_weight.T))
        #a_attention_scores = F.leaky_relu(a_edge_attr * self.a_att_weight)
        #r_attention_scores = F.leaky_relu(r_edge_attr * self.r_att_weight)
        
        a_attention_scores = F.leaky_relu(a_edge_attr * self.a_att_weight)
        r_attention_scores = F.leaky_relu(r_edge_attr * self.r_att_weight)
        
        #a_attention_weights = torch.softmax(a_attention_scores, dim=0)
        #r_attention_weights = torch.softmax(r_attention_scores, dim=0)
        
        a_attention_weights = a_attention_scores
        r_attention_weights = r_attention_scores

        # Integrate attention weights into normalization factor
        norm = norm  + a_attention_weights.view(-1)  +  r_attention_weights.view(-1)
             
        #print(type(norm))
        #print(norm)
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    
class TempAttnGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        in_channels = 64
        out_channels = 64 
        heads = 3
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # Define the learnable parameters for the attention mechanism
        self.lin_src = torch.nn.Linear(in_channels, heads * out_channels)
        self.lin_dst = torch.nn.Linear(in_channels, heads * out_channels)
        self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.a_att_weights = torch.nn.Parameter(torch.Tensor(heads, 1))
        self.r_att_weights = torch.nn.Parameter(torch.Tensor(heads, 1))
        self.reset_parameters()

    def reset_parameters(self):        
        torch.nn.init.xavier_uniform_(self.lin_src.weight)
        torch.nn.init.xavier_uniform_(self.lin_dst.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.xavier_uniform_(self.a_att_weights)
        torch.nn.init.xavier_uniform_(self.r_att_weights)
        
    def forward(self, x, edge_index, edge_attrs, size=None):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        H, C = self.heads, self.out_channels

        # Apply linear transformation to node embeddings
        x_i = self.lin_dst(x).view(-1, H, C)
        x_j = self.lin_src(x).view(-1, H, C)

        # Calculate alpha vectors for central nodes and neighbor nodes
        alpha_dst = torch.stack([(each_x_i * self.att_dst).sum(dim=1) for each_x_i in x_i])
        alpha_src = torch.stack([(each_x_j * self.att_src).sum(dim=1) for each_x_j in x_j])

        # Compute attention weights from edge attributes
        a_edge_attr = edge_attrs[:, 0].unsqueeze(1)
        r_edge_attr = edge_attrs[:, 1].unsqueeze(1)

        a_attention_scores = [F.leaky_relu(a_edge_attr * self.a_att_weights[i]) for i in range(self.heads)]
        r_attention_scores = [F.leaky_relu(r_edge_attr * self.r_att_weights[i]) for i in range(self.heads)]

        a_attention_weights = torch.softmax(sum(a_attention_scores), dim=0)
        r_attention_weights = torch.softmax(sum(r_attention_scores), dim=0)

        # Pass alpha and edge attributes as parameters to propagate
        out = self.propagate(edge_index, x=(x_i, x_j), alpha=(alpha_src, alpha_dst),
                             a_att_weights=a_attention_weights, r_att_weights=r_attention_weights, norm=norm, size=size).view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, a_att_weights, r_att_weights, norm):
        # Compute normalization
        norm = norm * a_att_weights.view(-1) * r_att_weights.view(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = alpha * norm.view(-1, 1)
        
        return alpha * x_j

    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")

class TempAttnLGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        in_channels = 64
        out_channels = 64 
        heads = 1
        
        self.count = 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # Define the learnable parameters for the attention mechanism
        self.a_att_weights = torch.nn.Parameter(torch.Tensor(heads, in_channels))
        self.r_att_weights = torch.nn.Parameter(torch.Tensor(heads, in_channels))

        print(f'TempAttnLGCN: in_channels={in_channels}, out_channels={out_channels}, heads={heads}')
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.a_att_weights)
        torch.nn.init.xavier_uniform_(self.r_att_weights)

    def forward(self, x, edge_index, edge_attrs, size=None):        
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Compute attention scores from edge attributes
        a_edge_attr = edge_attrs[:, 0].unsqueeze(1)
        r_edge_attr = edge_attrs[:, 1].unsqueeze(1)

        a_attention_scores = [F.leaky_relu(a_edge_attr * self.a_att_weights[i]) for i in range(self.heads)]
        r_attention_scores = [F.leaky_relu(r_edge_attr * self.r_att_weights[i]) for i in range(self.heads)]

        # Sum the attention scores from all heads
        a_attention_scores = torch.cat(a_attention_scores, dim=1)
        r_attention_scores = torch.cat(r_attention_scores, dim=1)
        
        # Compute attention weights using softmax over neighbors
        attention_scores = a_attention_scores + r_attention_scores
        attention_weights = geo_softmax(attention_scores, to_, num_nodes=x.size(0))
        
        #a_attention_weights = torch.softmax(sum(a_attention_scores), dim=0)
        #r_attention_weights = torch.softmax(sum(r_attention_scores), dim=0)
        
        #a_attention_weights = sum(a_attention_scores)
        #r_attention_weights = sum(r_attention_scores)

        # Pass the edge attributes as parameters to propagate
        #out = self.propagate(edge_index, x=x, a_att_weights=a_attention_weights, r_att_weights=r_attention_weights, norm=norm, size=size)
        out = self.propagate(edge_index, x=x, attention_weights=attention_weights, norm=norm, size=size)
        
        return out

    def message(self, x_j, attention_weights, norm):
        # Compute the final attention score for each edge
        
        #x_j = x_j * a_att_weights * r_att_weights
        weighted_messages = attention_weights * x_j
        
        #if self.count < 5:
        #  print(f'x_j={x_j.shape}, edge_attr={edge_attr.shape}, norm={norm.shape}, a_att_weights={a_att_weights.shape}, r_att_weights={r_att_weights.shape}')
        #  self.count += 1

        return norm.view(-1, 1) * weighted_messages


    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")


class TempAggAttnLGCN(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
        
        self.in_channels = 64
        self.out_channels = 16 
        self.heads = 1
        
        self.count = 0

        # Define the learnable parameters for the attention mechanism
        self.att_weights = torch.nn.Parameter(torch.Tensor(self.heads, self.in_channels))

        print(f'TempAggAttnLGCN: in_channels={self.in_channels}, out_channels={self.out_channels}, heads={self.heads}')
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att_weights)
        
    def forward(self, x, edge_index, edge_attrs, size=None):        
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Compute attention scores from edge attributes
        #edge_attr = edge_attrs.unsqueeze(1)

        att_scores = [F.leaky_relu(edge_attrs.unsqueeze(1) * self.att_weights[i]) for i in range(self.heads)]
   
        # Sum the attention scores from all heads
        att_scores = torch.cat(att_scores, dim=1)
        
        # Compute attention weights using softmax over neighbors
        att_weights = geo_softmax(att_scores, to_, num_nodes=x.size(0))
        
        if self.count < 3:
          print(f"attention_scores={att_scores.shape}, from_={from_.shape}, to_={to_.shape}, num_nodes={x.size(0)}")
          self.count += 1
        
        # Pass the edge attributes as parameters to propagate
        #out = self.propagate(edge_index, x=x, a_att_weights=a_attention_weights, r_att_weights=r_attention_weights, norm=norm, size=size)
        out = self.propagate(edge_index, x=x, attention_weights=att_weights, norm=norm, size=size)
        
        return out

    def message(self, x_j, attention_weights, norm):
        #x_j = x_j * a_att_weights * r_att_weights
        weighted_messages = attention_weights * x_j
        
        return norm.view(-1, 1) * weighted_messages
      
      
def chunked_matrix_multiplication(x, y, chunk_size):
    """
    Performs matrix multiplication in chunks to avoid CUDA out of memory errors.
    
    Args:
    - x (torch.Tensor): Tensor of shape (3276876, 1)
    - y (torch.Tensor): Tensor of shape (1, 3276876)
    - chunk_size (int): Size of chunks to use for the computation
    
    Returns:
    - result (torch.Tensor): Resulting tensor of shape (3276876, 3276876)
    """
    n = x.size(0)
    result = torch.zeros((n, n), device=x.device)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            result[i:end_i, j:end_j] = torch.matmul(x[i:end_i], y[:, j:end_j])
    
    return result
        
class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
            
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        #norm = gcn_norm(edge_index=edge_index, add_self_loops=False)
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm, attr = edge_attrs)

    def message(self, x_j, norm, attr):
        #return norm.view(-1, 1) * (x_j / attr.view(-1, 1))
        return norm.view(-1, 1) * (x_j / torch.sigmoid(attr).view(-1, 1))
        #return norm.view(-1, 1) * (x_j / torch.pow(attr, 20).view(-1, 1))
        #return norm.view(-1, 1) * (x_j * torch.exp(attr).view(-1, 1))
        #return norm.view(-1, 1) * (x_j / torch.exp(attr).view(-1, 1))
        #return norm.view(-1, 1) * x_j 
        #return norm.view(-1, 1) * (x_j / torch.sqrt(attr).view(-1, 1))
        #return norm.view(-1, 1) * (x_j / torch.log(attr).view(-1, 1))


    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    
class LightGCNConv2(MessagePassing):
    def __init__(self, **kwargs):  
        super().__init__(aggr='add')
            
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        #norm_tuple = gcn_norm(edge_index=edge_index, add_self_loops=False)
        #norm = norm_tuple[0]
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")

class NGCFConv(MessagePassing):
  def __init__(self, latent_dim, dropout, bias=True, **kwargs):  
    super(NGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout

    self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
    self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

    self.init_parameters()


  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)


  def forward(self, x, edge_index, edge_attrs):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)


  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) 


class NGCFConv2(MessagePassing):
  def __init__(self, latent_dim, dropout, bias=True, **kwargs):  
    super(NGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout

    self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
    self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

    self.init_parameters()


  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)


  def forward(self, x, edge_index, edge_attrs):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm, attr = edge_attrs)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)


  def message(self, x_j, x_i, norm, attr):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i)) * attr.view(-1, 1)
  
class RecSysGNN(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model, # 'NGCF' or 'LightGCN' or 'LightAttGCN'
      dropout=0.1, # Only used in NGCF
      is_temp=False
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'LightGCN' or model == 'LightAttnGCN' or model == 'TempLGCN' or model == 'TempAttnLGCN' or model == 'TempAttnGCN' or model == 'TempAggAttnLGCN'), \
        'Model must be NGCF or LightGCN or LightAttnGCN or TempLGCN or TempAttnLGCN or TempAttnGCN or TempAggAttnLGCN'
    self.model = model
    self.n_users = num_users
    self.n_items = num_items
    
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)
    
    
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(
        NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'LightGCN':
      self.convs = nn.ModuleList(
        LightGCNConv2() for _ in range(num_layers)
      )
    elif self.model == 'LightAttnGCN':
      self.convs = nn.ModuleList(
        LightAttnGCN() for _ in range(num_layers)
      )
    elif self.model == 'TempLGCN':
      self.convs = nn.ModuleList(
        TempLGCN() for _ in range(num_layers)
      )
    elif self.model == 'TempAttnLGCN':
      self.convs = nn.ModuleList(
        TempAttnLGCN() for _ in range(num_layers)
      )
    elif self.model == 'TempAttnGCN':
      self.convs = nn.ModuleList(
        TempAttnGCN() for _ in range(num_layers)
      )
    elif self.model == 'TempAggAttnLGCN': # TempAggAttnLGCN
      self.convs = nn.ModuleList(
        TempAggAttnLGCN() for _ in range(num_layers)
      )
    else:
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

    self.init_parameters()


  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1) 

  def forward(self, edge_index, edge_attrs):
    emb0 = self.embedding.weight
    embs = [emb0]

    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb, edge_index=edge_index, edge_attrs=edge_attrs)
      embs.append(emb)
      
    out = (
      torch.cat(embs, dim=-1) if self.model == 'NGCF' 
      else torch.mean(torch.stack(embs, dim=0), dim=0)
    )
        
    return emb0, out


  def encode_minibatch(self, users, pos_items, neg_items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)
    
    return (
        out[users], 
        out[pos_items], 
        out[neg_items],
        emb0[users],
        emb0[pos_items],
        emb0[neg_items]
    )

  def generate_unique_ids(self, user_ids, item_ids):
    """
    Generate unique IDs for user-item pairs.
    
    Parameters:
    user_ids (list or pd.Series): List or Series of user IDs.
    item_ids (list or pd.Series): List or Series of item IDs.
    num_items (int): Total number of distinct items (M).
    
    Returns:
    pd.Series: Series of unique IDs.
    """
    assert len(user_ids) == len(item_ids), 'user and item numbers must be the same'
    
    # I have this issue: TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    unique_ids = pd.Series(user_ids.cpu().numpy()) * self.n_items + pd.Series(item_ids.cpu().numpy())
    
    #unique_ids = pd.Series(user_ids) * self.n_items + pd.Series(item_ids)
    return unique_ids
    
    
class RecSysGNN2(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model, # 'NGCF' or 'LightGCN' or 'LightAttGCN'
      dropout=0.1, # Only used in NGCF
      is_temp=False
  ):
    super(RecSysGNN2, self).__init__()

    assert (model == 'NGCF' or model == 'LightGCN'), \
        'Model must be NGCF or LightGCN'
    self.model = model
    self.n_users = num_users
    self.n_items = num_items
    
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)
      
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(
        NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    else:
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

    self.init_parameters()


  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1) 
     
  def forward(self, edge_index, edge_attrs):
    emb0 = self.embedding.weight
    embs = [emb0]

    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb, edge_index=edge_index, edge_attrs=edge_attrs)
      embs.append(emb)
      
    out = (
      torch.cat(embs, dim=-1) if self.model == 'NGCF' 
      else torch.mean(torch.stack(embs, dim=0), dim=0)
    )
    
    return emb0, out


  def encode_minibatch(self, users, pos_items, neg_items, edge_index, edge_attrs):
    emb0, out = self(edge_index, edge_attrs)

    return (
        out[users], 
        out[pos_items], 
        out[neg_items],
        emb0[users],
        emb0[pos_items],
        emb0[neg_items]
    )
