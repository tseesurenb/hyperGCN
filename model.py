'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
from torch import nn, Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import degree, softmax as geo_softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm

        
class LightGCNAttn(MessagePassing):
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
        #return norm.view(-1, 1) * x_j   
        #return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        return norm.view(-1, 1) * (x_j * torch.sigmoid(attr).view(-1, 1))
        #return norm.view(-1, 1) * (x_j * torch.exp(attr).view(-1, 1))
        #return norm.view(-1, 1) * (x_j * torch.pow(attr, 20).view(-1, 1))
        #return norm.view(-1, 1) * (x_j * torch.log(attr).view(-1, 1))


    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    
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

    assert (model == 'NGCF' or model == 'LightGCN') or model == 'LightGCNAttn', 'Model must be NGCF or LightGCN or LightGCNAttn'
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
        LightGCNConv() for _ in range(num_layers)
      )
    elif self.model == 'LightGCNAttn':
      self.convs = nn.ModuleList(LightGCNAttn() for _ in range(num_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or LightAttGCN')

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