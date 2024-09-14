'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import torch_scatter
from torch import nn, Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import degree, softmax as geo_softmax
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

        
class LightGCNAttn(MessagePassing):
    def __init__(self, weight_mode = 'exp', **kwargs):  
        super().__init__(aggr='add')
        
        self.weight_mode = weight_mode
        self.graph_norms = None
        self.edge_attr_calc = None
            
    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        #from_, to_ = edge_index
        #deg = degree(to_, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        if self.graph_norms is None:
            self.graph_norms = gcn_norm(edge_index=edge_index, add_self_loops=False)
            
            print(f"Graph norms ({self.graph_norms.shape}): {self.graph_norms}")
            print(f"Edge attrs ({edge_attrs.shape}): {edge_attrs}")
            print(f"Edge index ({edge_index.shape}): {edge_index}")
            
            if self.weight_mode == 'exp':
                self.edge_attr_calc = torch.exp(edge_attrs)
            elif self.weight_mode == 'raw':
                self.edge_attr_calc = edge_attrs
            else:
                self.edge_attr_calc = None
        
        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=self.graph_norms[0], attr = self.edge_attr_calc)

    def message(self, x_j, norm, attr):
        if attr != None:
            return norm.view(-1, 1) * (x_j * attr.view(-1, 1))
        else:
            return norm.view(-1, 1) * x_j   
        

    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="sum")
    
    #def aggregate(self, x, messages, index):
    #    return torch_scatter.scatter(messages, index, self.node_dim, reduce="mean")
    

class LightGCNAttn2(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):  
        super().__init__(aggr='add')
        self.att = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attrs):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages with attention
        return self.propagate(edge_index, x=x, norm=norm, edge_attrs=edge_attrs)

    def message(self, x_i, x_j, edge_attrs, norm):
        # Compute attention coefficients
        edge_attr = edge_attrs.unsqueeze(-1) if edge_attrs.dim() == 1 else edge_attrs
        alpha = F.leaky_relu((x_i * self.att).sum(dim=-1)) + F.leaky_relu((x_j * self.att).sum(dim=-1))
        alpha = F.softmax(alpha, dim=0)
        
        return alpha.view(-1, 1) * norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # Apply linear transformation
        return self.linear(aggr_out)

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


class GraphSage(MessagePassing):
    
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):  #  __init__(self, in_channels, out_channels, args, **kwargs):
      
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = latent_dim
        self.out_channels = latent_dim
        self.normalize = True
        bias = bias
        self.lin_src = None
        self.lin_dst = None

        ############# Your code here #############
        # Define the layers needed for the message and update functions below.
        # self.lin_src is the linear transformation that you apply to aggregated 
        #            message from neighbors.
        # self.lin_dst is the linear transformation that  you apply to embedding 
        #            for central node.
        # Our implementation is ~2 lines, but don't worry if you deviate from this.
        self.lin_src = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_dst = nn.Linear(latent_dim, latent_dim, bias=bias)
        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(self, x, edge_index, edge_attrs, size = None):
        """"""

        out = None

        ############# Your code here #############
        # Implement message passing, as well as any post-processing (our update rule).
        # 1. Call the propagate function to conduct message passing.
        #    1.1 See the description of propagate above or the following link for more information: 
        #        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        #    1.2 You will only use the representation for neighbor nodes (x_j) in message passing. 
        #        Thus, you can simply pass the same representation for src / dst as x=(x, x). 
        #        Although we give this to you, try thinking through what this means following
        #        the descriptions above.
        # 2. Update your node embeddings with a skip connection.
        # 3. If normalize is set, do L-2 normalization (defined in 
        #    torch.nn.functional)
        #
        # Our implementation is ~5 lines, but don't worry if you deviate from this. 
        out = self.propagate(edge_index, x=(x, x), size=size)
        out = self.lin_dst(x) + out
        #out = F.relu(out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        ############################################################################ 

        return out

    def message(self, x_j):

        out = None

        ############# Your code here #############
        # Implement your message function here.
        # Hint: Look at the formulation of t he mean aggregation function, focusing on 
        # what message each individual neighboring node passes during aggregation.
        #
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out = self.lin_src(x_j) #/ x_j.size(0)
        ############################################################################ 

        return out

    def aggregate(self, inputs, index, dim_size = None):

        out = None

        # The axis along which to index number of nodes.
        node_dim = self.node_dim

        ############# Your code here #############
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: 
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        #
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean')
        ############################################################################

        return out


class GAT(MessagePassing):

    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = latent_dim
        self.out_channels = latent_dim
        self.heads = 1
        self.negative_slope = 0.2
        self.dropout = 0.5
        bias = False
        self.debug_g = True

        self.lin_src = None
        self.lin_dst = None
        self.att_src = None
        self.att_dst = None

        ############# Your code here #############
        # Define the layers needed for the message functions below.
        # self.lin_src is the linear transformation that you apply to embeddings 
        # BEFORE message passing.
        # 
        # Pay attention to dimensions of the linear layers, especially when
        # implementing multi-head attention.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        self.lin_src = nn.Linear(latent_dim, latent_dim * self.heads, bias=bias)
        ############################################################################

        self.lin_dst = self.lin_src

        ############# Your code here #############
        # Define the attention parameters \overrightarrow{a_{src}/{dst}}^T in the above intro.
        # 1. Be mindful of when you want to include multi-head attention.
        # 2. Note that for each attention head we parametrize the attention parameters 
        #    as weight vectors NOT matrices - i.e. their first dimension should be 1.
        self.att_src = nn.Parameter(torch.Tensor(self.heads, latent_dim))
        self.att_dst = nn.Parameter(torch.Tensor(self.heads,  latent_dim))
        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_attrs, size = None):
        
        H, C = self.heads, self.out_channels

        ############# Your code here #############
        # Implement message passing, as well as any pre- and post-processing (our update rule).
        # 1. First apply linear transformation to node embeddings, and split that 
        #    into multiple heads. We use the same representations for source and
        #    target nodes, but apply different linear weights (W_{src} and W_{dst})
        # 2. Calculate alpha vectors for central nodes (alpha_{dst}) and neighbor nodes (alpha_{src}).
        # 3. Call propagate function to conduct the message passing. 
        #    3.1 Remember to pass alpha = (alpha_{src}, alpha_{dst}) as a parameter.
        #    3.2 See here for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # 4. Transform the output back to the shape of N * d.
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        x_i = self.lin_dst(x).view(-1, H, C)
        x_j = self.lin_src(x).view(-1, H, C)
        alpha_dst = torch.stack([(each_x_i * self.att_dst).sum(dim=1) for each_x_i in x_i])
        alpha_src = torch.stack([(each_x_j * self.att_src).sum(dim=1) for each_x_j in x_j])
        out = self.propagate(edge_index, x=(x_i, x_j), alpha=(alpha_src, alpha_dst), size=size, attrs = edge_attrs).view(-1, H * C)
        ############################################################################

        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, attrs):

        ############# Your code here #############
        # Implement your message function. Putting the attention in message 
        # instead of in update is a little tricky.
        # 1. Calculate the attention weights using alpha_i and alpha_j,
        #    and apply leaky ReLU.
        # 2. Calculate softmax over the neighbor nodes for all the nodes. Use 
        #    torch_geometric.utils.softmax instead of the one in Pytorch.
        # 3. Apply dropout to attention weights (alpha).
        # 4. Multiply embeddings and attention weights. As a sanity check, the output
        #    should be of shape (E, H, d).
        # 5. ptr (LongTensor, optional): If given, computes the softmax based on
        #    sorted inputs in CSR representation. You can simply pass it to softmax.
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        #attrs = attrs.to(torch.float32)
        alpha_ij = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        #alpha_ij = F.leaky_relu(alpha_i + alpha_j + attrs.view(-1, 1, 1), negative_slope=self.negative_slope)
        alpha_ij = softmax(alpha_ij, index, num_nodes=size_i)    
        alpha_ij = F.dropout(alpha_ij, p=self.dropout, training=self.training)
        out = x_j * alpha_ij.view(-1, self.heads, 1)        
        if ptr is not None:
            out = softmax(out, ptr = ptr)
        ############################################################################
        return out
    def aggregate(self, inputs, index, dim_size = None):

        ############# Your code here #############
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html
        # Pay attention to "reduce" parameter is different from that in GraphSage.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out = torch_scatter.scatter(inputs, index, dim=0, reduce='sum')
        ############################################################################
    
        return out
  
class RecSysGNN(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model, # 'NGCF' or 'LightGCN' or 'LightAttGCN'
      dropout=0.1, # Only used in NGCF
      is_temp=False,
      weight_mode = None,
      base = False
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'LightGCN') or model == 'LightGCNAttn' or model == 'GraphSage' or model == 'GAT', 'Model must be NGCF or LightGCN or LightGCNAttn or GraphSage or GAT'
    self.model = model
    self.n_users = num_users
    self.n_items = num_items
    self.base = base
    
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)
    
    if self.base:
      self.b_embedding = nn.Embedding(num_users + num_items, latent_dim)
    
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(
        NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'LightGCN':
      self.convs = nn.ModuleList(
        LightGCNConv() for _ in range(num_layers)
      )
    elif self.model == 'GraphSage':
      self.convs = nn.ModuleList(
        GraphSage(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'GAT':
      self.convs = nn.ModuleList(
        GAT(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'LightGCNAttn':
      self.convs = nn.ModuleList(LightGCNAttn(weight_mode=weight_mode) for _ in range(num_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or LightAttGCN')

    self.init_parameters()

  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1) 
      
      if self.base:
        nn.init.normal_(self.b_embedding.weight, std=0.1)

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
    
    if self.base:
      u_base = self.b_embedding.weight[users]
      pos_items_base = self.b_embedding.weight[pos_items]
      neg_items_base = self.b_embedding.weight[neg_items]
      
      return (
          out[users], 
          out[pos_items], 
          out[neg_items],
          u_base,
          pos_items_base,
          neg_items_base,
          emb0[users],
          emb0[pos_items],
          emb0[neg_items]
      )
    else:
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
  
class RecSysGNN_2(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model, # 'NGCF' or 'LightGCN' or 'LightAttGCN'
      dropout=0.1, # Only used in NGCF
      is_temp=False,
      weight_mode = None
  ):
    super(RecSysGNN_2, self).__init__()

    assert (model == 'NGCF' or model == 'LightGCN') or model == 'LightGCNAttn' or model == 'GraphSage' or model == 'GAT', 'Model must be NGCF or LightGCN or LightGCNAttn or GraphSage or GAT'
    self.model = model
    self.n_users = num_users
    self.n_items = num_items
    
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)
    
    self.b_embedding = nn.Embedding(num_users + num_items, latent_dim)
    
    if self.model == 'NGCF':
      self.convs = nn.ModuleList(
        NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'LightGCN':
      self.convs = nn.ModuleList(
        LightGCNConv() for _ in range(num_layers)
      )
    elif self.model == 'GraphSage':
      self.convs = nn.ModuleList(
        GraphSage(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'GAT':
      self.convs = nn.ModuleList(
        GAT(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    elif self.model == 'LightGCNAttn':
      self.convs = nn.ModuleList(LightGCNAttn(weight_mode=weight_mode) for _ in range(num_layers))
    else:
      raise ValueError('Model must be NGCF, LightGCN or LightAttGCN')

    self.init_parameters()


  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      # Authors of LightGCN report higher results with normal initialization
      nn.init.normal_(self.embedding.weight, std=0.1) 
      nn.init.normal_(self.b_embedding.weight, std=0.1)

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
    
    u_base = self.b_embedding[users],
    pos_items_base = self.b_embedding[pos_items],
    neg_items_base = self.b_embedding[neg_items],
        
    return (
        out[users], 
        out[pos_items], 
        out[neg_items],
        u_base,
        pos_items_base,
        neg_items_base,
        emb0[users],
        emb0[pos_items],
        emb0[neg_items]
    )