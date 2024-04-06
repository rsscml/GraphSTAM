#!/usr/bin/env python
# coding: utf-8

# Model Specific imports

import torch
import copy
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, Linear, to_hetero, HeteroConv, GCNConv, SAGEConv, GraphConv, HeteroDictLinear
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import gc

#from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAE, TweedieLoss, PoissonLoss, MAPE, SMAPE

# Data specific imports

from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

# core data imports
import pandas as pd
import numpy as np
import itertools
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# utilities imports
from joblib import Parallel, delayed
import shutil

# #### Models & Utils

# Generic Layer to allow directionality consideration in any MPNN layer. Currently not released in PyG

class DirGNNConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on directed
    graphs as described in the `"Edge Directionality Improves Learning on
    Heterophilic Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.
    :class:`DirGNNConv` will pass messages both from source nodes to target
    nodes and from target nodes to source nodes.

    Args:
        conv (MessagePassing): The underlying
            :class:`~torch_geometric.nn.conv.MessagePassing` layer to use.
        alpha (float, optional): The alpha coefficient used to weight the
            aggregations of in- and out-edges as part of a convex combination.
            (default: :obj:`0.5`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will add
            transformed root node features to the output.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        conv: MessagePassing,
        alpha: float = 0.5,
        root_weight: bool = True,):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)

        if hasattr(conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = Linear(conv.in_channels, conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        r""""""
        x_in = self.conv_in(x, edge_index)
        x_out = self.conv_out(x, edge_index.flip([0]))

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'
    
# additional helper functions from: https://github.com/emalgorithm/directed-graph-neural-network for paper: https://arxiv.org/abs/2305.10498  
    
def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_adj(edge_index, num_nodes, graph_type="directed"):
    """
    Return the type of adjacency matrix specified by `graph_type` as sparse tensor.
    """
    if graph_type == "transpose":
        edge_index = torch.stack([edge_index[1], edge_index[0]])
    elif graph_type == "undirected":
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    elif graph_type == "directed":
        pass
    else:
        raise ValueError(f"{graph_type} is not a valid graph type")

    value = torch.ones((edge_index.size(1),), device=edge_index.device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes))


def compute_unidirectional_edges_ratio(edge_index):
    num_directed_edges = edge_index.shape[1]
    num_undirected_edges = torch_geometric.utils.to_undirected(edge_index).shape[1]

    num_unidirectional = num_undirected_edges - num_directed_edges

    return (num_unidirectional / (num_undirected_edges / 2)) * 100


# Causal Masked Attention

class MaskedCausalAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, size, device):
        super().__init__()
        
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        self.attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
    
    def forward(self, node_embeddings):
        attn_out, _ = self.multihead_attn(query=node_embeddings, 
                                          key=node_embeddings, 
                                          value=node_embeddings, 
                                          attn_mask=self.attn_mask, 
                                          is_causal=True)
        return attn_out
        

# loss function

class QuantileLoss():
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(self, quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]):
        self.quantiles = quantiles

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses
      
class RMSE():
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self):
        super().__init__()

    def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:
        loss = torch.pow(y_pred - target, 2)
        return loss


# Reference implementation from the DirGNN paper: https://arxiv.org/abs/2305.10498 

class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x)


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x) + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads, concat=True)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads, concat=True)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(x, edge_index_t)
    

class DirGATv2Conv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATv2Conv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATv2Conv(input_dim, output_dim, heads=heads, concat=True)
        self.conv_dst_to_src = GATv2Conv(input_dim, output_dim, heads=heads, concat=True)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(x, edge_index_t)
    


# Forecast GNN Layers

class HeteroForecastSageConv(torch.nn.Module):
    
    def __init__(self, 
                 hidden_channels,  
                 edge_types, 
                 node_types, 
                 target_node_type,
                 context_node_type,
                 num_layers=1,
                 alpha=0.5,
                 use_linear_pretransform=True,
                 aggr='mean',
                 skip_connection=True,
                 use_dirgnn=True):
        
        super().__init__()
        
        self.target_node_type = target_node_type
        self.context_node_type = context_node_type
        self.use_linear_pretransform = use_linear_pretransform
        self.skip_connection = skip_connection
        
        if self.use_linear_pretransform:
            self.linear_layers = torch.nn.ModuleList()
            lin_dict = HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=node_types)
            self.linear_layers.append(lin_dict)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for e in edge_types:
                if (e[0] == e[2]):
                    if use_dirgnn:
                        #mpnn = SAGEConv(in_channels=-1, out_channels=hidden_channels)
                        #conv_dict[e] = DirGNNConv(conv = mpnn, alpha = alpha, root_weight = True)
                        conv_dict[e] = DirSageConv(input_dim=-1, output_dim=hidden_channels, alpha=alpha)
                    else:
                        conv_dict[e] = SAGEConv(in_channels=(-1,-1), out_channels=hidden_channels)
                elif (e[0] in self.context_node_type) or (e[2] in self.context_node_type):
                    # global context nodes
                    conv_dict[e] = SAGEConv(in_channels=(-1,-1), out_channels=hidden_channels)
                else:
                    if i == 0:
                        conv_dict[e] = SAGEConv(in_channels=(-1,-1), out_channels=hidden_channels)
                    else:
                        # layers after first layer operate only on demand nodes & not covariates
                        pass
                    
            conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        
        if self.use_linear_pretransform:
            # linear transform node features
            for lin_dict in self.linear_layers:
                x_dict = lin_dict(x_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        if self.skip_connection:        
            res_x_dict = x_dict
            #print("res_x_dict: ",res_x_dict.keys())

        # apply dir sage layer to transformed dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.skip_connection:
                #print("x_dict: ",x_dict.keys())
                res_x_dict = {key: res_x_dict[key] for key in x_dict.keys()}
                #print("res_x_dict: ",res_x_dict.keys())
                x_dict = {key: x + res_x for (key, x), (res_key, res_x)  in zip(x_dict.items(), res_x_dict.items()) if key == res_key}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            #print("x_dict: ",x_dict.keys())

        out = self.lin(x_dict[self.target_node_type])

        return out 
    

class HeteroForecastGCNConv(torch.nn.Module):
    
    def __init__(self, 
                 hidden_channels,  
                 edge_types, 
                 node_types, 
                 target_node_type,
                 context_node_type,
                 num_layers=1,
                 alpha=0.5,
                 use_linear_pretransform=True,
                 aggr='mean',
                 skip_connection=True,
                 use_dirgnn=True):
        
        super().__init__()
        
        self.target_node_type = target_node_type
        self.context_node_type = context_node_type
        self.use_linear_pretransform = use_linear_pretransform
        self.skip_connection = skip_connection
        
        if self.use_linear_pretransform:   
            self.linear_layers = torch.nn.ModuleList()
            lin_dict = HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=node_types)
            self.linear_layers.append(lin_dict)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for e in edge_types:
                if e[0] == e[2]:
                    if use_dirgnn:
                        #mpnn = GCNConv(in_channels=-1, out_channels=hidden_channels)
                        #conv_dict[e] = DirGNNConv(conv = mpnn, alpha = alpha, root_weight = True)
                        conv_dict[e] = DirGCNConv(input_dim=-1, output_dim=hidden_channels, alpha=alpha)
                    else:
                        conv_dict[e] = GCNConv(in_channels=-1, out_channels=hidden_channels)
                elif (e[0] in self.context_node_type) or (e[2] in self.context_node_type):
                    # global context nodes
                    conv_dict[e] = SAGEConv(in_channels=(-1,-1), out_channels=hidden_channels)
                else:
                    if i==0:
                        conv_dict[e] = SAGEConv(in_channels=(-1,-1), out_channels=hidden_channels)
                    else:
                        pass
                    
            conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        
        if self.use_linear_pretransform: 
            # linear transform node features
            for lin_dict in self.linear_layers:
                x_dict = lin_dict(x_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                
        if self.skip_connection:        
            res_x_dict = x_dict 
        
        # apply dir sage layer to transformed dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.skip_connection:
                res_x_dict = {key: res_x_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x)  in zip(x_dict.items(), res_x_dict.items()) if key == res_key}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        out = self.lin(x_dict[self.target_node_type])

        return out 
    

class HeteroForecastGATConv(torch.nn.Module):
    
    def __init__(self, 
                 hidden_channels,  
                 edge_types, 
                 node_types, 
                 target_node_type,
                 context_node_type,
                 heads=1,
                 num_layers=1,
                 alpha=0.5,
                 use_linear_pretransform=True,
                 aggr='mean',
                 skip_connection=True,
                 use_dirgnn=True):
        
        super().__init__()
        
        self.target_node_type = target_node_type
        self.context_node_type = context_node_type
        self.use_linear_pretransform = use_linear_pretransform
        self.skip_connection = skip_connection
        
        if self.use_linear_pretransform:  
            self.linear_layers = torch.nn.ModuleList()
            lin_dict = HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=node_types)
            self.linear_layers.append(lin_dict)
   
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for e in edge_types:
                if e[0] == e[2]:
                    if use_dirgnn:
                        conv_dict[e] = DirGATConv(input_dim=-1, output_dim=int(hidden_channels/heads), heads=heads, alpha=alpha)
                    else:
                        conv_dict[e] = GATConv(in_channels=-1, out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=True)
                elif (e[0] in self.context_node_type) or (e[2] in self.context_node_type):
                    # global context nodes
                    conv_dict[e] = GATConv(in_channels=(-1,-1), out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=False)
                else:
                    if i == 0:
                        conv_dict[e] = GATConv(in_channels=(-1,-1), out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=False)
                    else:
                        pass
                    
            conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        
        if self.use_linear_pretransform:  
            # linear transform node features
            for lin_dict in self.linear_layers:
                x_dict = lin_dict(x_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                
        if self.skip_connection:        
            res_x_dict = x_dict 

        # apply dir sage layer to transformed dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.skip_connection:
                res_x_dict = {key: res_x_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x)  in zip(x_dict.items(), res_x_dict.items()) if key == res_key}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        out = self.lin(x_dict[self.target_node_type])

        return out 

    
class HeteroForecastGATv2Conv(torch.nn.Module):
    
    def __init__(self, 
                 hidden_channels,  
                 edge_types, 
                 node_types, 
                 target_node_type,
                 context_node_type,
                 heads=1,
                 num_layers=1,
                 alpha=0.5,
                 use_linear_pretransform=True,
                 aggr='mean',
                 skip_connection=True,
                 use_dirgnn=True):
        
        super().__init__()
        
        self.target_node_type = target_node_type
        self.context_node_type = context_node_type
        self.use_linear_pretransform = use_linear_pretransform
        self.skip_connection = skip_connection
        
        if self.use_linear_pretransform:  
            self.linear_layers = torch.nn.ModuleList()
            lin_dict = HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=node_types)
            self.linear_layers.append(lin_dict)
   
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for e in edge_types:
                if e[0] == e[2]:
                    if use_dirgnn:
                        conv_dict[e] = DirGATv2Conv(input_dim=-1, output_dim=int(hidden_channels/heads), heads=heads, alpha=alpha)
                    else:
                        conv_dict[e] = GATv2Conv(in_channels=-1, out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=True)
                elif (e[0] in self.context_node_type) or (e[2] in self.context_node_type):
                    # global context nodes
                    conv_dict[e] = GATv2Conv(in_channels=(-1,-1), out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=False)
                else:
                    if i == 0:
                        conv_dict[e] = GATv2Conv(in_channels=(-1,-1), out_channels=int(hidden_channels/heads), heads=heads, concat=True, add_self_loops=False)
                    else:
                        pass
                    
            conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        
        if self.use_linear_pretransform:  
            # linear transform node features
            for lin_dict in self.linear_layers:
                x_dict = lin_dict(x_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        if self.skip_connection:        
            res_x_dict = x_dict 

        # apply dir sage layer to transformed dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.skip_connection:
                res_x_dict = {key: res_x_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x)  in zip(x_dict.items(), res_x_dict.items()) if key == res_key}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        out = self.lin(x_dict[self.target_node_type])

        return out 

# Models

class STGNN(torch.nn.Module):
    
    def __init__(self,
                 model_type,
                 model_option,
                 hidden_channels, 
                 out_channels, 
                 metadata, 
                 n_nodes, 
                 target_node,
                 context_nodes,
                 seq_len,
                 device,
                 heads=1,
                 n_quantiles=1, 
                 num_layers=1,
                 lstm_layers=1,
                 alpha=0.5, 
                 dropout=0.0,
                 residual_conn_type='concat',
                 positive_output=False,
                 aggr='mean',
                 use_linear_pretransform=True,
                 apply_norm_layers=True,
                 skip_connection=True,
                 use_dirgnn=True):
        
        """
        model options:
        
        model_type: ['SAGE','GCN','GAT','GATV2']
        model_options: ['BASIC','LSTM','LSTMAttention','Attention']
        loss_type: ['Point','Quantile']
        positive_output: True/False
        residual_conn_type; ['add','concat']
        
        """
        
        super(STGNN, self).__init__()
        
        self.model_type = str.upper(model_type)
        self.model_option = str.upper(model_option)
        
        self.hidden_channels = hidden_channels
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.target_node = target_node
        self.context_nodes = context_nodes
        self.seq_len = seq_len
        self.device = device
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.alpha = alpha
        self.dropout = dropout
        self.pos_out = positive_output
        self.n_quantiles = n_quantiles
        self.num_layers = num_layers
        self.lstm_layers = lstm_layers
        self.apply_norm_layers = apply_norm_layers
        self.use_dirgnn = use_dirgnn
        self.skip_connection = skip_connection
        
        # for LSTM layers
        self.residual_conn_type = residual_conn_type
        
        if self.residual_conn_type == 'add':
            self.lstm_input_dim = hidden_channels
        elif self.residual_conn_type == 'concat':
            self.lstm_input_dim = int(2*hidden_channels)
        
        self.lstm_output_dim = self.lstm_input_dim
        
        # for Attention Layers
        
        self.aggr = aggr
        self.use_linear_pretransform = use_linear_pretransform
        
        if self.model_type == "SAGE":
            self.gnn_layer = HeteroForecastSageConv(hidden_channels = self.hidden_channels,  
                                                    edge_types = self.edge_types, 
                                                    node_types = self.node_types, 
                                                    target_node_type = self.target_node,
                                                    context_node_type = self.context_nodes,
                                                    num_layers = self.num_layers,
                                                    alpha = self.alpha,
                                                    use_linear_pretransform = self.use_linear_pretransform,
                                                    aggr = self.aggr,
                                                    skip_connection = self.skip_connection,
                                                    use_dirgnn = self.use_dirgnn)
        elif self.model_type == "GCN":
            self.gnn_layer = HeteroForecastGCNConv(hidden_channels = self.hidden_channels,  
                                                    edge_types = self.edge_types, 
                                                    node_types = self.node_types, 
                                                    target_node_type = self.target_node,
                                                    context_node_type = self.context_nodes,
                                                    num_layers = self.num_layers,
                                                    alpha = self.alpha,
                                                    use_linear_pretransform = self.use_linear_pretransform,
                                                    aggr = self.aggr,
                                                    skip_connection = self.skip_connection,
                                                    use_dirgnn = self.use_dirgnn)
        elif self.model_type == "GAT":
            self.gnn_layer = HeteroForecastGATConv(hidden_channels = self.hidden_channels,  
                                                    edge_types = self.edge_types, 
                                                    node_types = self.node_types, 
                                                    target_node_type = self.target_node,
                                                    context_node_type = self.context_nodes,
                                                    heads = self.heads,
                                                    num_layers = self.num_layers,
                                                    alpha = self.alpha,
                                                    use_linear_pretransform = self.use_linear_pretransform,
                                                    aggr = self.aggr,
                                                    skip_connection = self.skip_connection,
                                                    use_dirgnn = self.use_dirgnn)
            
        elif self.model_type == "GATV2":
            self.gnn_layer = HeteroForecastGATv2Conv(hidden_channels = self.hidden_channels,  
                                                    edge_types = self.edge_types, 
                                                    node_types = self.node_types, 
                                                    target_node_type = self.target_node,
                                                    context_node_type = self.context_nodes,
                                                    heads = self.heads,
                                                    num_layers = self.num_layers,
                                                    alpha = self.alpha,
                                                    use_linear_pretransform = self.use_linear_pretransform,
                                                    aggr = self.aggr,
                                                    skip_connection = self.skip_connection,
                                                    use_dirgnn = self.use_dirgnn)
            
        else:
            raise "Invalid GNN Layer Specified. Valid Layers: [GAT, GATV2, GCN, SAGE] "
            
        if self.model_option == "BASIC":
            # direct projection from node embeddings
            self.layer_norm1 = torch.nn.LayerNorm(self.hidden_channels)
            self.project_lin = Linear(self.hidden_channels, self.n_pred*self.n_quantiles)
        
        elif self.model_option == "LSTM":
            # add lstm stack
            self.layer_norm1 = torch.nn.LayerNorm(self.hidden_channels)
            self.layer_norm2 = torch.nn.LayerNorm(self.hidden_channels)
            self.lstm_stack = torch.nn.LSTM(input_size = self.lstm_input_dim, 
                                            hidden_size = self.lstm_output_dim, 
                                            num_layers = self.lstm_layers, 
                                            batch_first = True)
            
            self.layer_norm3 = torch.nn.LayerNorm(self.lstm_output_dim)
            self.target_transform = torch.nn.Linear(1, self.hidden_channels)
            self.project_lin = Linear(self.lstm_output_dim, self.n_pred*self.n_quantiles)
        
        elif self.model_option == "LSTMATTENTION":
            #add lstm stack
            self.layer_norm1 = torch.nn.LayerNorm(self.hidden_channels)
            self.layer_norm2 = torch.nn.LayerNorm(self.hidden_channels)
            self.lstm_stack = torch.nn.LSTM(input_size = self.lstm_input_dim, 
                                            hidden_size = self.lstm_output_dim, 
                                            num_layers = self.lstm_layers, 
                                            batch_first = True)
            
            self.layer_norm3 = torch.nn.LayerNorm(self.lstm_output_dim)
            # add attention layer
            self.multihead_attn = MaskedCausalAttention(embed_dim = self.lstm_output_dim, 
                                                        heads = self.heads, 
                                                        size = self.seq_len, 
                                                        device = self.device)
            
            self.layer_norm4 = torch.nn.LayerNorm(self.lstm_output_dim)
            self.target_transform = torch.nn.Linear(1, self.hidden_channels)
            self.project_lin = Linear(self.lstm_output_dim, self.n_pred*self.n_quantiles)
        
        elif self.model_option == "ATTENTION":
            # add attention layer
            self.layer_norm1 = torch.nn.LayerNorm(self.hidden_channels)
            self.layer_norm2 = torch.nn.LayerNorm(self.hidden_channels)
            self.multihead_attn = MaskedCausalAttention(embed_dim = self.hidden_channels, 
                                                        heads = self.heads, 
                                                        size = self.seq_len, 
                                                        device = self.device)
            
            self.layer_norm3 = torch.nn.LayerNorm(self.hidden_channels)
            self.target_transform = torch.nn.Linear(1, self.hidden_channels)
            self.project_lin = Linear(self.hidden_channels, self.n_pred*self.n_quantiles)
        
        else:
            raise "Invalid model_option. model_option: [BASIC, LSTM, LSTMATTENTION, ATTENTION]"
            
    def forward(self, x_dict, edge_index_dict):
    
        # gnn layer
        x = self.gnn_layer(x_dict, edge_index_dict)
        x = x.relu()
        
        if self.apply_norm_layers:
            x = self.layer_norm1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # output preds now without temporal attention
        
        if self.model_option == "BASIC":
            
            node_embed_batches = []
            for i in range(self.seq_len):
                emb_batch_step = x[int(i*self.n_nodes):int((i+1)*self.n_nodes)]
                emb_batch_step = torch.unsqueeze(emb_batch_step, dim=1)
                node_embed_batches.append(emb_batch_step)

            node_embed_seq = torch.concat(node_embed_batches, dim=1)
    
            # final projection layer
            out = self.project_lin(node_embed_seq[:,-1,:])
            
            if self.pos_out:
                out = F.softplus(out)

            if self.n_quantiles > 1:
                out = torch.reshape(out, (-1, self.n_pred, self.n_quantiles))
            else:
                out = torch.reshape(out, (-1, self.n_pred))

        elif self.model_option == "LSTM":
            
            node_embed_batches = []
            for i in range(self.seq_len):
                emb_batch_step = x[int(i*self.n_nodes):int((i+1)*self.n_nodes)]
                emb_batch_step = torch.unsqueeze(emb_batch_step, dim=1)
                node_embed_batches.append(emb_batch_step)

            node_embed_seq = torch.concat(node_embed_batches, dim=1)
            
            # reshape original sequence (for residual connection)
            res_x = self.target_transform(x_dict[self.target_node][:, -1:])
            res_x = res_x.relu()
            if self.apply_norm_layers:
                res_x = self.layer_norm2(res_x)

            res_batches = []
            for i in range(self.seq_len):
                res_batch_step = res_x[int(i*self.n_nodes):int((i+1)*self.n_nodes), :]
                res_batch_step = torch.unsqueeze(res_batch_step, dim=1)
                res_batches.append(res_batch_step)   
            
            res_batch_seq = torch.concat(res_batches, dim=1)

            # Apply LSTM stack with residual conn
            if self.residual_conn_type == 'add':
                input_seq = torch.add(node_embed_seq, res_batch_seq)/2
            else:
                input_seq = torch.concat([node_embed_seq, res_batch_seq], dim=2)
                 
            lstm_out, _ = self.lstm_stack(input_seq)
            lstm_out = lstm_out.relu()
            
            if self.apply_norm_layers:
                lstm_out = self.layer_norm3(lstm_out)   

            # final projection layer
            out = self.project_lin(lstm_out[:,-1,:])
            
            if self.pos_out:
                out = F.softplus(out)

            if self.n_quantiles > 1:
                out = torch.reshape(out, (-1, self.n_pred, self.n_quantiles))
            else:
                out = torch.reshape(out, (-1, self.n_pred))

        elif self.model_option == "LSTMATTENTION":
            
            node_embed_batches = []
            for i in range(self.seq_len):
                emb_batch_step = x[int(i*self.n_nodes):int((i+1)*self.n_nodes)]
                emb_batch_step = torch.unsqueeze(emb_batch_step, dim=1)
                node_embed_batches.append(emb_batch_step)

            node_embed_seq = torch.concat(node_embed_batches, dim=1)
            
            # reshape original sequence (for residual connection)
            res_x = self.target_transform(x_dict[self.target_node][:, -1:])
            res_x = res_x.relu()
            if self.apply_norm_layers:
                res_x = self.layer_norm2(res_x)

            res_batches = []
            for i in range(self.seq_len):
                res_batch_step = res_x[int(i*self.n_nodes):int((i+1)*self.n_nodes), :]
                res_batch_step = torch.unsqueeze(res_batch_step, dim=1)
                res_batches.append(res_batch_step)   
            
            res_batch_seq = torch.concat(res_batches, dim=1)

            # Apply LSTM stack with residual conn
            if self.residual_conn_type == 'add':
                input_seq = torch.add(node_embed_seq, res_batch_seq)/2
            else:
                input_seq = torch.concat([node_embed_seq, res_batch_seq], dim=2)
                   
            lstm_out, _ = self.lstm_stack(input_seq)
            lstm_out = lstm_out.relu()
            
            if self.apply_norm_layers:
                lstm_out = self.layer_norm3(lstm_out)
            
            # Apply Attention
            attn_out = self.multihead_attn(lstm_out)
            attn_out = attn_out.relu()
            
            if self.apply_norm_layers:
                attn_out = self.layer_norm4(attn_out)

            # final projection layer
            out = self.project_lin(attn_out[:,-1,:])
            
            if self.pos_out:
                out = F.softplus(out)

            if self.n_quantiles > 1:
                out = torch.reshape(out, (-1, self.n_pred, self.n_quantiles))
            else:
                out = torch.reshape(out, (-1, self.n_pred))
        
        elif self.model_option == "ATTENTION":
            
            node_embed_batches = []
            for i in range(self.seq_len):
                emb_batch_step = x[int(i*self.n_nodes):int((i+1)*self.n_nodes)]
                emb_batch_step = torch.unsqueeze(emb_batch_step, dim=1)
                node_embed_batches.append(emb_batch_step)

            node_embed_seq = torch.concat(node_embed_batches, dim=1)
            
            # reshape original sequence (for residual connection)
            res_x = self.target_transform(x_dict[self.target_node][:, -1:])
            res_x = res_x.relu()
            if self.apply_norm_layers:
                res_x = self.layer_norm2(res_x)

            res_batches = []
            for i in range(self.seq_len):
                res_batch_step = res_x[int(i*self.n_nodes):int((i+1)*self.n_nodes), :]
                res_batch_step = torch.unsqueeze(res_batch_step, dim=1)
                res_batches.append(res_batch_step)   
            
            res_batch_seq = torch.concat(res_batches, dim=1)

            # Apply Attention with residual conn
            if self.residual_conn_type == 'add':
                input_seq = torch.add(node_embed_seq, res_batch_seq)/2
            else:
                input_seq = torch.concat([node_embed_seq, res_batch_seq], dim=2)
                
            # Apply Attention
            attn_out = self.multihead_attn(input_seq)
            attn_out = attn_out.relu()
            
            if self.apply_norm_layers:
                attn_out = self.layer_norm3(attn_out)

            # final projection layer
            out = self.project_lin(attn_out[:,-1,:])
            
            if self.pos_out:
                out = F.softplus(out)

            if self.n_quantiles > 1:
                out = torch.reshape(out, (-1, self.n_pred, self.n_quantiles))
            else:
                out = torch.reshape(out, (-1, self.n_pred))

        return out


# #### Graph Data Generator

class graphmodel():
    def __init__(self, 
                 col_dict, 
                 max_history,
                 max_lags,
                 max_leads,
                 train_till,
                 test_till,
                 fh = 1,
                 batch = 1, 
                 scaling_method = 'mean_scaling',
                 categorical_onehot_encoding = True,
                 directed_graph = True,
                 create_all_temporal_edges = False,
                 include_rolling_features = True,
                 rolling_window_size = 13,
                 shuffle = True,
                 interleave = 1,
                 PARALLEL_DATA_JOBS = 4, 
                 PARALLEL_DATA_JOBS_BATCHSIZE = 128):
        """
        col_dict: dictionary of various column groups {id_col:'',
                                                       target_col:'',
                                                       time_index_col:'',
                                                       global_context_col_list:[],
                                                       static_cat_col_list:[],
                                                       temporal_known_num_col_list:[],
                                                       temporal_unknown_num_col_list:[],
                                                       temporal_known_cat_col_list:[],
                                                       temporal_unknown_cat_col_list:[],
                                                       strata_col_list:[] # UNUSED
                                                       sort_col_list:[] # UNUSED
                                                       wt_col:None # UNUSED
                                                       }
        window_len: history_len + forecast_horizon
        fh: forecast_horizon
        batch: batch_size (per strata)
        min_nz: min. no. of non-zeros in the target input series to be eligible for train/test batch
        scaling_method: 'mean_scaling','no_scaling'
        
        """
        super().__init__()
        
        self.col_dict = copy.deepcopy(col_dict)
        self.fh = int(fh)
        self.max_history = int(max_history)
        self.max_lags = int(max_lags) if (max_lags is not None) and (max_lags>0) else 1
        self.max_leads = int(max_leads) if (max_leads is not None) and (max_leads>0) else 1
        self.rolling_window_size = rolling_window_size
        self.snap_sequence_len = int(max(self.max_lags,self.rolling_window_size)+self.max_leads+1)
        self.window_len = int(self.max_history+self.max_leads+max(self.max_lags,self.rolling_window_size))
        
        assert self.max_leads >= self.fh, "max_leads must be >= fh"
        
        # adjust train_till/test_till for delta|max_leads - fh| in split_* methods
        self.train_till = train_till
        self.test_till = test_till
        
        self.batch = batch
        self.scaling_method = scaling_method
        self.categorical_onehot_encoding = categorical_onehot_encoding
        self.directed_graph = directed_graph
        self.create_all_temporal_edges = create_all_temporal_edges
        self.include_rolling_features = include_rolling_features
        self.shuffle = shuffle
        self.interleave = interleave
        self.PARALLEL_DATA_JOBS = PARALLEL_DATA_JOBS
        self.PARALLEL_DATA_JOBS_BATCHSIZE = PARALLEL_DATA_JOBS_BATCHSIZE
        
        self.pad_constant = 0 #if self.scaling_method == 'mean_scaling' else -1
       
        # extract columnsets from col_dict
        self.id_col = self.col_dict.get('id_col', None)
        self.target_col = self.col_dict.get('target_col', None)
        self.time_index_col = self.col_dict.get('time_index_col', None)
        self.datetime_format = self.col_dict.get('datetime_format', None)
        self.strata_col_list = self.col_dict.get('strata_col_list', [])
        self.sort_col_list = self.col_dict.get('sort_col_list', [self.id_col, self.time_index_col])
        self.wt_col = self.col_dict.get('wt_col', None)
        self.global_context_col_list = self.col_dict.get('global_context_col_list', [])
        self.static_cat_col_list = self.col_dict.get('static_cat_col_list', [])
        self.temporal_known_num_col_list = self.col_dict.get('temporal_known_num_col_list', [])
        self.temporal_unknown_num_col_list = self.col_dict.get('temporal_unknown_num_col_list', [])
        self.temporal_known_cat_col_list = self.col_dict.get('temporal_known_cat_col_list', [])
        self.temporal_unknown_cat_col_list = self.col_dict.get('temporal_unknown_cat_col_list', [])

        if (self.id_col is None) or (self.target_col is None) or (self.time_index_col is None):
            raise ValueError("Id Column, Target Column or Index Column not specified!")

        # full columnset for train/test/infer
        self.col_list = [self.id_col] + [self.target_col] + \
                         self.static_cat_col_list + self.global_context_col_list + \
                         self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                         self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.cat_col_list = self.global_context_col_list + self.static_cat_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
        
        self.context_cols = []
        for col in self.global_context_col_list:
            self.context_cols.append(f'snap_id_{col}')
        

    def scale_dataset(self, df):
        """
        Individually scale each 'id' & concatenate them all in one dataframe. Uses Joblib for parallelization.
        """
        # filter out ids with insufficient timestamps (at least one datapoint should be before train cutoff period)
        df = df.groupby(self.id_col).filter(lambda x: x[self.time_index_col].min()<self.train_till)

        groups = df.groupby([self.id_col])
        scaled_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(self.df_scaler)(gdf) for _, gdf in groups)
        gdf = pd.concat(scaled_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
       
        return gdf
    

    def df_scaler(self, gdf):
        """
        Scales a dataframe based on the chosen scaling method & columns specification 
        """
        # obtain scalers
        
        scale_gdf = gdf[gdf[self.time_index_col]<=self.train_till].reset_index(drop=True)
        
        if self.scaling_method == 'mean_scaling':
            target_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.target_col])), 1.0)
            target_sum = np.sum(np.abs(scale_gdf[self.target_col]))
            scale = np.divide(target_sum, target_nz_count) + 1.0
            
            if len(self.temporal_known_num_col_list) > 0:
                known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
                known_scale = np.divide(known_sum, known_nz_count) + 1.0
            else:
                known_scale = 1

            if len(self.temporal_unknown_num_col_list) > 0:
                unknown_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0), 1.0)
                unknown_sum = np.sum(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0)
                unknown_scale = np.divide(unknown_sum, unknown_nz_count) + 1.0
            else:
                unknown_scale = 1

        elif self.scaling_method == 'standard_scaling':
            scale_mu = scale_gdf[self.target_col].mean()
            scale_std = np.maximum(scale_gdf[self.target_col].std(), 0.0001)
            scale = [scale_mu, scale_std]

            if len(self.temporal_known_num_col_list) > 0:
                known_mean = np.mean(scale_gdf[self.temporal_known_num_col_list].values, axis=0)
                known_stddev = np.maximum(np.std(scale_gdf[self.temporal_known_num_col_list].values, axis=0), 0.0001)
                known_scale = [known_mean, known_stddev]
            else:
                known_scale = [0, 1]

            if len(self.temporal_unknown_num_col_list) > 0:
                unknown_mean = np.mean(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0)
                unknown_stddev = np.maximum(np.std(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0), 0.0001)
                unknown_scale = [unknown_mean, unknown_stddev]
            else:
                unknown_scale = [0, 1]

        elif self.scaling_method == 'no_scaling':
            scale = 1.0
            known_scale = 1.0
            unknown_scale = 1.0
        
        # reset index
        gdf = gdf.reset_index(drop=True)
        
        # scale each feature independently
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            gdf[self.target_col] = gdf[self.target_col]/scale
            gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list]/known_scale
            gdf[self.temporal_unknown_num_col_list] = gdf[self.temporal_unknown_num_col_list]/unknown_scale
        
        elif self.scaling_method == 'standard_scaling':
            gdf[self.target_col] = (gdf[self.target_col] - scale[0])/scale[1]
            gdf[self.temporal_known_num_col_list] = (gdf[self.temporal_known_num_col_list] - known_scale[0])/known_scale[1]
            gdf[self.temporal_unknown_num_col_list] = (gdf[self.temporal_unknown_num_col_list] - unknown_scale[0])/unknown_scale[1]
        
        # Store scaler as a column
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            gdf['scaler'] = scale
        elif self.scaling_method == 'standard_scaling':
            gdf['scaler_mu'] = scale[0]
            gdf['scaler_std'] = scale[1]
        
        return gdf
    
    def scale_labelencodings(self, df):
        """
        1. Create & store label encodings
        2. Max scale label encoded columns across dataset
        """
        try:
            # reuse encoders for inference
            for col in self.label_encoded_col_list:
                df[col] = self.label_encoders[col].fit_transform(df[col])
                df[col] = df[col]/max(df[col].max(), 1)
        except:
            self.label_encoders = {}
            for col in self.label_encoded_col_list:
                self.label_encoders[col] = preprocessing.LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
                df[col] = df[col]/max(df[col].max(), 1)

        return df
    
    def sort_dataset(self, data):
        """
        sort pandas dataframe by provided col list & order
        """
        if len(self.sort_col_list) > 0:
            data = data.sort_values(by=self.sort_col_list, ascending=True)
        else:
            pass
        return data
    
    
    def check_null(self, data):
        """
        Check for columns containing NaN
        """
        null_cols = []
        null_status = None
        for col in self.col_list:
            if data[col].isnull().any():
                null_cols.append(col)
        
        if len(null_cols)>0:
            null_status == True
        else:
            null_status == False
        
        return null_status, null_cols
    

    def onehot_encode(self, df):
        
        # onehot encode all cat cols
        # exclude static & global cols as they use numeric & dummy feature later on
        #df = pd.concat([df[self.cat_col_list], pd.get_dummies(data=df, columns=self.cat_col_list, prefix_sep='_')], axis=1, join='inner')
        
        onehot_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + self.global_context_col_list
        df = pd.concat([df[onehot_col_list], pd.get_dummies(data=df, columns=onehot_col_list, prefix_sep='_')], axis=1, join='inner')
        
        return df
    
    def get_roll_stats(self, df):
        
        # for each static col, get common target_col stats (moving average, wma, stddev etc.)
        #rolling_stat_window_size = int(self.window_len - self.max_leads)
        
        rolling_stat_window_size = self.rolling_window_size
        
        self.rolling_stat_cols = []
        for col in [self.id_col]:
            df[f'{col}_rollsum'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).sum())
            df[f'{col}_rollmean'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).mean())
            df[f'{col}_rollstd'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).std())
            df[f'{col}_rollqtile50'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).quantile(0.50))
            df[f'{col}_rollqtile75'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).quantile(0.75))
            df[f'{col}_rollqtile90'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).quantile(0.90))
            df[f'{col}_rollqtile97'] = df.groupby(col)[self.target_col].transform(lambda x: x.rolling(rolling_stat_window_size, 1).quantile(0.97))
            
            # add cols to group self.temporal_unknown_num_col_list
            self.temporal_unknown_num_col_list.append(f'{col}_rollsum')
            self.temporal_unknown_num_col_list.append(f'{col}_rollmean')
            self.temporal_unknown_num_col_list.append(f'{col}_rollstd')
            self.temporal_unknown_num_col_list.append(f'{col}_rollqtile50')
            self.temporal_unknown_num_col_list.append(f'{col}_rollqtile75')
            self.temporal_unknown_num_col_list.append(f'{col}_rollqtile90')
            self.temporal_unknown_num_col_list.append(f'{col}_rollqtile97')
            
            # to keep track of rolling stat cols
            self.rolling_stat_cols.append(f'{col}_rollsum')
            self.rolling_stat_cols.append(f'{col}_rollmean')
            self.rolling_stat_cols.append(f'{col}_rollstd')
            self.rolling_stat_cols.append(f'{col}_rollqtile50')
            self.rolling_stat_cols.append(f'{col}_rollqtile75')
            self.rolling_stat_cols.append(f'{col}_rollqtile90')
            self.rolling_stat_cols.append(f'{col}_rollqtile97')
        
        return df
            
    def pad_dataframe(self, df):
        # this ensures num nodes in a graph don't change from period to period. Essentially, we introduce dummy nodes.
        
        # store original dtypes for re-casting later
        #original_dtypes = df.dtypes.to_dict()
        
        # function to fill NaNs in group id & stat cols post padding
        def fillgrpid(x):
            id_val = x[self.id_col].unique().tolist()[0]
            x = dateindex.merge(x, on=[self.time_index_col], how='left').fillna({self.id_col: id_val})
            
            for col in self.global_context_col_list + self.global_context_onehot_cols:
                x[col] = x[col].fillna(method='ffill')
                x[col] = x[col].fillna(method='bfill')
            
            for col in self.static_cat_col_list:
                x[col] = x[col].fillna(method='ffill')
                x[col] = x[col].fillna(method='bfill')
            
            if self.categorical_onehot_encoding:
                for col in self.known_onehot_cols:
                    x[col] = x[col].fillna(0)

                for col in self.unknown_onehot_cols:
                    x[col] = x[col].fillna(0)
            else:
                # use -1 to signal padding since label encoding will be from 0,...,n
                for col in self.label_encoded_col_list:
                    x[col] = x[col].fillna(-1)
                
            # add mask
            x['y_mask'] = np.where(x[self.target_col].isnull(), 0, 1)
            
            # pad target col
            if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
                x[self.target_col] = x[self.target_col].fillna(0)
            elif self.scaling_method == 'standard_scaling':
                pad_value = -(x['scaler_mean'].unique().tolist()[0])/(x['scaler_std'].unique().tolist()[0])
                x[self.target_col] = x[self.target_col].fillna(pad_value)
            
            return x   

        # get a df of all timestamps in the dataset
        dateindex = pd.DataFrame(sorted(df[self.time_index_col].unique()), columns=[self.time_index_col]) 
        
        # "padded" dataset with padding constant used as nan filler
        df = df.groupby(self.id_col, sort=False).apply(lambda x: fillgrpid(x).fillna(self.pad_constant)).reset_index(drop=True)
        
        # add target mask
        #df['y_mask'] = np.where(df[self.target_col]==self.pad_constant, 0, 1)
        
        # align datatypes within columns; some columns may have mixed types as a result of pad_constant
        for col, datatype in df.dtypes.to_dict().items():
            if col != 'Unnamed: 0':
                if datatype == 'object':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(datatype)
                    
        # convert all bool columns to 1/0 (problem on linux only)
        #print("converting bool to int")
        
        for col in self.known_onehot_cols+self.unknown_onehot_cols:
            if df[col].dtypes.name == 'bool':
                df[col] = df[col]*1.0
            else:
                df[col] = df[col].astype(str).astype(bool).astype(int)
        
        return df

    def get_node_features_label(self):

        self.node_features_label = {}

        for col in [self.target_col] + self.rolling_stat_cols + self.temporal_known_num_col_list + self.known_onehot_cols:

            if col in [self.target_col] + self.rolling_stat_cols:
                features = [f'{col}_lag_{i}' for i in range(self.snap_sequence_len-(self.max_leads-self.fh) - self.fh, 0, -1)]
                self.node_features_label[col] = features + self.rolling_stat_cols
            else:
                lag_features = [f'{col}_lag_{i}' for i in range(self.snap_sequence_len - self.max_leads, 0, -1)]
                lead_features = [f'{col}_lead_{i}' for i in range(1, self.max_leads+1)]
                self.node_features_label[col] = lag_features + lead_features

    def preprocess(self, data):
        
        print("   preprocessing dataframe - check for null columns...")
        # check null
        null_status, null_cols = self.check_null(data)
        
        if null_status:
            print("NaN column(s): ", null_cols)
            raise ValueError("Column(s) with NaN detected!")
            
        # sort
        print("   preprocessing dataframe - sort by datetime & id...")
        df = self.sort_dataset(data)
        
        # obtain rolling stats
        if self.include_rolling_features:
            print("   preprocessing dataframe - get rolling stats by group...")
            df = self.get_roll_stats(df)
        else:
            self.rolling_stat_cols = []
        
        # scale dataset
        print("   preprocessing dataframe - scale numeric cols...")
        df = self.scale_dataset(df)
       
        # onehot encode
        if self.categorical_onehot_encoding:
            print("   preprocessing dataframe - onehot encode categorical columns...")
            df = self.onehot_encode(df)
        else:
            print("   preprocessing dataframe - label encode & scale categorical columns...")
            # use label encoding as numeric feature
            self.temporal_known_num_col_list += self.temporal_known_cat_col_list
            self.temporal_unknown_num_col_list += self.temporal_unknown_cat_col_list
            # keep record of label encoded columns for padding in a later step
            self.label_encoded_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
            # empty the cat col lists
            self.temporal_known_cat_col_list = []
            self.temporal_unknown_cat_col_list = []
            # create & scale labelencodings
            df = self.scale_labelencodings(df)
            
        print("   preprocessing dataframe - gather node specific feature cols...")
        # node types & node features
        self.node_cols = [self.target_col] + self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
        
        self.node_features = {}
        self.known_onehot_cols = []
        self.unknown_onehot_cols = []
        self.global_context_onehot_cols = []

        for node in self.global_context_col_list:
            # one-hot col names
            onehot_cols_prefix = str(node) + '_'
            onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
            self.node_features[node] = onehot_col_features
            self.global_context_onehot_cols += onehot_col_features

        for node in self.node_cols:
            if node not in self.cat_col_list:
                self.node_features[node] = [node]
            elif node in self.temporal_known_cat_col_list:
                # one-hot col names
                onehot_cols_prefix = str(node)+'_' 
                onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
                self.node_features[node] = onehot_col_features
                self.known_onehot_cols += onehot_col_features
            elif node in self.temporal_unknown_cat_col_list:
                # one-hot col names
                onehot_cols_prefix = str(node)+'_' 
                onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
                self.node_features[node] = onehot_col_features
                self.unknown_onehot_cols += onehot_col_features

        self.temporal_nodes =  self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list 

        # get node_features_label
        self.get_node_features_label()

        return df
    
    def node_indexing(self, df, node_cols):
        # hold the indices <-> col_value map in a dict
        col_id_map = {}
        for col in node_cols:
            # Sort to define the order of nodes
            node_sorted_df = df[[col]]
            # drop duplicates
            node_sorted_df = node_sorted_df.drop_duplicates()
            node_sorted_df = node_sorted_df.sort_values(by=col).set_index(col)
            # Map IDs to start from 0
            node_sorted_df = node_sorted_df.reset_index(drop=False)
            node_indices = node_sorted_df[col]
            node_id_map = node_indices.reset_index().set_index(col).to_dict()
            col_id_map[col] = node_id_map
            
        return col_id_map
    
    def squeeze_dataframe(self, df_snap):
        
        if (self.max_leads - self.fh) >= 1: 
            df_target = df_snap.groupby(self.id_col, sort=False)[self.target_col].apply(lambda x: x[-self.snap_sequence_len:-(self.max_leads-self.fh)].values).reset_index()
            df_mask = df_snap.groupby(self.id_col, sort=False)['y_mask'].apply(lambda x: x[-self.snap_sequence_len:-(self.max_leads-self.fh)].values).reset_index(drop=True)
        else:
            df_target = df_snap.groupby(self.id_col, sort=False)[self.target_col].apply(lambda x: x[-self.snap_sequence_len:].values).reset_index()
            df_mask = df_snap.groupby(self.id_col, sort=False)['y_mask'].apply(lambda x: x[-self.snap_sequence_len:].values).reset_index(drop=True)
            
        known_num_arrays_df = []    
        for col in self.temporal_known_num_col_list:
            df_temp = df_snap.groupby(self.id_col, sort=False)[col].apply(lambda x: x[-self.snap_sequence_len:].values).reset_index(drop=True)
            known_num_arrays_df.append(df_temp)
        
        unknown_num_arrays_df = []    
        for col in self.temporal_unknown_num_col_list:
            df_temp = df_snap.groupby(self.id_col, sort=False)[col].apply(lambda x: x[-self.snap_sequence_len:-self.max_leads].values).reset_index(drop=True)
            unknown_num_arrays_df.append(df_temp)
            
        known_cat_arrays_df = []    
        for col in self.known_onehot_cols:
            df_temp = df_snap.groupby(self.id_col, sort=False)[col].apply(lambda x: x[-self.snap_sequence_len:].values).reset_index(drop=True)
            known_cat_arrays_df.append(df_temp)
            
        unknown_cat_arrays_df = []    
        for col in self.unknown_onehot_cols:
            df_temp = df_snap.groupby(self.id_col, sort=False)[col].apply(lambda x: x[-self.snap_sequence_len:-self.max_leads].values).reset_index(drop=True)
            unknown_cat_arrays_df.append(df_temp)
            
        #global_context_df = []    
        #for col in self.global_context_col_list:
        #    df_temp = df_snap.groupby(self.id_col, sort=False)[col].apply(lambda x: x.unique()[0]).reset_index(drop=True)
        #    global_context_df.append(df_temp)
        
        """
        self.static_numeric_feats = []
        all_ts = sorted(df_snap[self.time_index_col].unique().tolist())[:-self.max_leads]
        min_ts = all_ts[0]
        max_ts = all_ts[-1]
        df_hist = df_snap[(df_snap[self.time_index_col]<=max_ts)&(df_snap[self.time_index_col]>=min_ts)]
        for col in self.static_cat_col_list:
            df_hist[f'{col}_snapmean'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.mean(x))
            df_hist[f'{col}_snapstd'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.std(x))
            df_hist[f'{col}_snapqtile10'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.1))
            df_hist[f'{col}_snapqtile25'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.25))
            df_hist[f'{col}_snapqtile50'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.5))
            df_hist[f'{col}_snapqtile75'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.75))
            df_hist[f'{col}_snapqtile90'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.9))
            df_hist[f'{col}_snapqtile97'] = df_hist.groupby(col)[self.target_col].transform(lambda x: np.quantile(x, 0.97))
            self.static_numeric_feats.append(f'{col}_snapmean')
            self.static_numeric_feats.append(f'{col}_snapstd')
            self.static_numeric_feats.append(f'{col}_snapqtile10')
            self.static_numeric_feats.append(f'{col}_snapqtile25')
            self.static_numeric_feats.append(f'{col}_snapqtile50')
            self.static_numeric_feats.append(f'{col}_snapqtile75')
            self.static_numeric_feats.append(f'{col}_snapqtile90')
            self.static_numeric_feats.append(f'{col}_snapqtile97')
        
        """
        # merge by id_col
        df_list = [df_target, df_mask]+known_num_arrays_df+unknown_num_arrays_df+known_cat_arrays_df+unknown_cat_arrays_df
        squeezed_df = pd.concat(df_list, axis=1, join='inner')
        
        # add back un-onehot encoded stat columns for edge indices manipulation
        #static_df = df_hist[[self.id_col]+self.global_context_col_list+self.static_cat_col_list].drop_duplicates()
        
        static_df = df_snap[[self.id_col]+self.global_context_col_list+self.global_context_onehot_cols+self.static_cat_col_list].drop_duplicates()
        
        squeezed_df = squeezed_df.merge(static_df, on=self.id_col, how='inner')
        
        return squeezed_df
    
    
    def create_snapshot_graph(self, df_snap, period):
        
        # get squeezed dataframe for each timestep & stack vertically in order
        period_list = sorted(df_snap[self.time_index_col].unique().tolist())[self.snap_sequence_len-1:]
        num_periods = len(period_list)
        
        #print("num_periods: ", num_periods)
        #print("period_list: ", period_list)
        
        sqdf_list = []
        for i, t in enumerate(period_list):
            #
            df_t = df_snap[df_snap[self.time_index_col]<=t].copy()
            # get squeezed df
            sqdf_t = self.squeeze_dataframe(df_t)
            sqdf_t['current_snap'] = t
            if i == num_periods-1:
                sqdf_t['next_snap'] = period_list[i]
            else:
                sqdf_t['next_snap'] = period_list[i+1]
            
            sqdf_list.append(sqdf_t)
        
        sqdf = pd.concat(sqdf_list, axis=0)
        
        # create new snap node id
        sqdf['snap_id_col'] = sqdf[self.id_col].astype(str) + sqdf['current_snap'].astype(str)
        
        snap_global_context_col_list = []
        for col in self.global_context_col_list:
            sqdf[f'snap_id_{col}'] = sqdf[col].astype(str) + sqdf['current_snap'].astype(str)
            snap_global_context_col_list.append(f'snap_id_{col}')
            
        snap_static_cat_col_list = []
        for col in self.static_cat_col_list:
            sqdf[f'snap_id_{col}'] = sqdf[col].astype(str) + sqdf['current_snap'].astype(str)
            snap_static_cat_col_list.append(f'snap_id_{col}')
            
        # index 'id_col'
        col_map_dict = self.node_indexing(sqdf, ['snap_id_col', self.id_col]+snap_static_cat_col_list+snap_global_context_col_list)
        #print(col_map_dict)
        
        # map id to indices
        for col, id_map in col_map_dict.items():
            sqdf[col] = sqdf[col].map(id_map["index"]).astype(int)
            
        #print(sqdf.shape, sqdf[self.id_col].nunique(), sqdf['snap_id_col'].nunique())
        
        # Create HeteroData Object
        data = HeteroData({"y_mask":None})
        
        # get node features
        
        # target
        arr_list = sqdf[self.target_col].tolist()
        feats = np.concatenate(arr_list, axis=0).reshape(-1, self.snap_sequence_len-(self.max_leads-self.fh))
        data[self.target_col].x = torch.tensor(feats[:,:-self.fh], dtype=torch.float)
        data[self.target_col].y = torch.tensor(feats[:,-self.fh:], dtype=torch.float)
        
        # store snapshot period
        data[self.target_col].time_attr = period
        
        # add an output mask
        mask_arr_list = sqdf['y_mask'].tolist()
        mask_feat = np.concatenate(mask_arr_list, axis=0).reshape(-1, self.snap_sequence_len-(self.max_leads-self.fh))
        data[self.target_col].y_mask = torch.tensor(mask_feat[:,-self.fh:], dtype=torch.float)
        
        for col in self.temporal_known_num_col_list:
            arr_list = sqdf[col].tolist()
            feats = np.concatenate(arr_list, axis=0).reshape(-1, self.snap_sequence_len)
            data[col].x = torch.tensor(feats, dtype=torch.float)
            
        for col in self.temporal_unknown_num_col_list:
            arr_list = sqdf[col].tolist()
            feats = np.concatenate(arr_list, axis=0).reshape(-1, self.snap_sequence_len-self.max_leads)
            data[col].x = torch.tensor(feats, dtype=torch.float)
        
        for col in self.known_onehot_cols:
            arr_list = sqdf[col].tolist()
            feats = np.concatenate(arr_list, axis=0).reshape(-1, self.snap_sequence_len)
            data[col].x = torch.tensor(feats, dtype=torch.float)
        
        for col in self.unknown_onehot_cols:
            arr_list = sqdf[col].tolist()
            feats = np.concatenate(arr_list, axis=0).reshape(-1, self.snap_sequence_len-self.max_leads)
            data[col].x = torch.tensor(feats, dtype=torch.float)
            
        current_snaps_list = sorted(sqdf['current_snap'].unique().tolist())
        next_snaps_list = sorted(sqdf['next_snap'].unique().tolist())
        
        # global context node features (one-hot features)
        for col in snap_global_context_col_list:
            orig_col = col.lstrip('snap_id_')
            onehot_cols_prefix = str(orig_col) + '_'
            onehot_col_features = [c for c in sqdf.columns.tolist() if str(c).startswith(onehot_cols_prefix)]
            feats_df = sqdf[[col] + onehot_col_features].drop_duplicates()
            data[col].x = torch.tensor(feats_df[onehot_col_features].to_numpy(), dtype=torch.float)
            #feats_df = sqdf[[col]]
            #feats_df[f'dummy_global_{col}'] = 1  # assign a constant as dummy feature
            #feats_df = feats_df.drop_duplicates()
            #data[col].x = torch.tensor(feats_df[[f'dummy_global_{col}']].to_numpy(), dtype=torch.float)
                
        # bidirectional edges between global context node & target_col nodes
        for col in snap_global_context_col_list:
            col_unique_values = sorted(sqdf[col].unique().tolist())
            
            fwd_edges_stack = []
            rev_edges_stack = []
            for value in col_unique_values:
                for snap in current_snaps_list:
                    # get subset of all nodes with common col value
                    edges = sqdf[(sqdf[col]==value)&(sqdf['current_snap']==snap)][['snap_id_col',col]].to_numpy()
                    rev_edges = sqdf[(sqdf[col]==value)&(sqdf['current_snap']==snap)][[col,'snap_id_col']].to_numpy()
                    fwd_edges_stack.append(edges)
                    rev_edges_stack.append(rev_edges)
                    
            # fwd edges
            edges = np.concatenate(fwd_edges_stack, axis=0)
            edge_name = (self.target_col,'hascontext_{}'.format(col),col)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            # reverse edges
            rev_edges = np.concatenate(rev_edges_stack, axis=0)
            rev_edge_name = (col,'{}_contextof'.format(col),self.target_col)
            data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)
            
        # bidirectional edges exist between target_col nodes related by various static cols
        
        for col in self.static_cat_col_list:
            col_unique_values = sorted(sqdf[col].unique().tolist())
        
            fwd_edges_stack = []
            rev_edges_stack = []
            for value in col_unique_values:
                
                for snap in current_snaps_list:
                    # get subset of all nodes with common col value
                    nodes = sqdf[(sqdf[col]==value)&(sqdf['current_snap']==snap)]['snap_id_col'].to_numpy()
                    # Build all combinations of connected nodes
                    permutations = list(itertools.combinations(nodes, 2))
                    edges_source = [e[0] for e in permutations]
                    edges_target = [e[1] for e in permutations]
                    edges = np.column_stack([edges_source, edges_target])
                    rev_edges = np.column_stack([edges_target, edges_source])
                    fwd_edges_stack.append(edges)
                    rev_edges_stack.append(rev_edges)
                    
            # edge names
            edge_name = (self.target_col,'relatedby_{}'.format(col),self.target_col)
            rev_edge_name = (self.target_col,'rev_relatedby_{}'.format(col),self.target_col)
            # add edges to Data()
            edges = np.concatenate(fwd_edges_stack, axis=0)
            rev_edges = np.concatenate(rev_edges_stack, axis=0)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)
                 
        # static nodes only required in this kind of connection
        """
        for col in snap_static_cat_col_list:
            #feat_cols_prefix = str(col)+'_'
            #feat_col_features = [c for c in self.static_numeric_feats if str(c).startswith(feat_cols_prefix)]
            #feats_df = sqdf[[col]+feat_col_features].drop_duplicates()
            #data[col].x = torch.tensor(np.sum(feats_df[feat_col_features].to_numpy(), axis=1, keepdims=True), dtype=torch.float)
            feats_df = sqdf[[col]]
            feats_df[f'dummy_static_{col}'] = 1  # assign a constant as dummy feature
            feats_df = feats_df.drop_duplicates()
            data[col].x = torch.tensor(feats_df[[f'dummy_static_{col}']].to_numpy(), dtype=torch.float)
        """

        #for col in self.static_cat_col_list:
        #    # directional edges from static col to target nodes
        #    
        #    col_unique_values = sorted(sqdf[col].unique().tolist())
        #    fwd_edges_stack = []
        #    bwd_edges_stack = []
        #    for value in col_unique_values:
        #        
        #        for snap in current_snaps_list:
        #            # get subset of all nodes with common col value
        #            edges = sqdf[(sqdf[col]==value)& (sqdf['current_snap']==snap)][[col,'snap_id_col']].to_numpy()
        #            rev_edges = sqdf[(sqdf[col]==value)& (sqdf['current_snap']==snap)][['snap_id_col',col]].to_numpy()
        #            fwd_edges_stack.append(edges)
        #            bwd_edges_stack.append(rev_edges)
        #            
        #    edges = np.concatenate(fwd_edges_stack, axis=0)
        #    edge_name = (col,'{}_contains'.format(col),self.target_col)
        #    data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
        #    
        #    if not self.directed_graph:
        #        rev_edges = np.concatenate(bwd_edges_stack, axis=0)
        #        rev_edge_name = (self.target_col,'belongsto_'.format(col), col)
        #        data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)
                
                
        # directed edges are from covariates to target
        
        for col in self.temporal_known_num_col_list+self.temporal_unknown_num_col_list+self.known_onehot_cols+self.unknown_onehot_cols:
            
            all_edges_stack= []
            for snap in current_snaps_list:
                nodes = sqdf[(sqdf['current_snap']==snap)]['snap_id_col'].to_numpy()
                edges = np.column_stack([nodes, nodes])
                all_edges_stack.append(edges)
                
            edge_name = (col,'{}_effect'.format(col),self.target_col)
            # add edges to Data()
            edges = np.concatenate(all_edges_stack, axis=0)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            
            if not self.directed_graph:
                rev_edge_name = (self.target_col,'covar_embed_update_{}'.format(col),col)
                data[rev_edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
                
            
        # temporal (directed) edges from one snap_id_col to next
        
        if self.max_history > 1:
            if self.create_all_temporal_edges:
                temporal_edges_arrlist = []
                for i in range(1, self.max_history):
                    snapid_key_pairs = sqdf.groupby(self.id_col)['snap_id_col'].apply(lambda x: list(zip(x.tolist(), x.tolist()[i:])))
                    snapid_edges = np.concatenate([np.array(tup).reshape(1,2) for tup in snapid_key_pairs.explode().values.tolist()], axis=0)
                    temporal_edges_arrlist.append(snapid_edges)
                temporal_edges = np.concatenate(temporal_edges_arrlist, axis=0)    
                snapid_edge_name = (self.target_col,'{}_temporal_effect'.format(self.target_col),self.target_col)
                data[snapid_edge_name].edge_index = torch.tensor(temporal_edges.transpose(), dtype=torch.long)
            else:
                snapid_key_pairs = sqdf.groupby(self.id_col)['snap_id_col'].apply(lambda x: list(zip(x.tolist(), x.tolist()[1:])))
                snapid_edges = np.concatenate([np.array(tup).reshape(1,2) for tup in snapid_key_pairs.explode().values.tolist()], axis=0)
                snapid_edge_name = (self.target_col,'{}_temporal_effect'.format(self.target_col),self.target_col)
                data[snapid_edge_name].edge_index = torch.tensor(snapid_edges.transpose(), dtype=torch.long)
        else:
            # no temporal edges applicable here
            pass

        # temporal edges from prev covar nodes to next snap_id_col
        
        """
        if self.max_history > 1:
            for col in self.temporal_known_num_col_list+self.temporal_unknown_num_col_list+self.known_onehot_cols+self.unknown_onehot_cols:
                snapid_key_pairs = sqdf.groupby(self.id_col)['snap_id_col'].apply(lambda x: list(zip(x.tolist(), x.tolist()[1:])))
                snapid_edges = np.concatenate([np.array(tup).reshape(1,2) for tup in snapid_key_pairs.explode().values.tolist()], axis=0)
            
                snapid_edge_name = (col,'{}_temporal_effect'.format(col),self.target_col)
                data[snapid_edge_name].edge_index = torch.tensor(snapid_edges.transpose(), dtype=torch.long)
        else:
            pass
        """

        # validate dataset
        print("validate snapshot graph ...")    
        data.validate(raise_on_error=True)
        
        return data
    
    def create_train_test_dataset(self, df):
        # preprocess
        print("preprocessing dataframe...")
        df = self.preprocess(df)
        
        # pad dataframe if required (will return df unchanged if not)
        print("padding dataframe...")
        df = self.pad_dataframe(df)
        
        # split into train,test,infer
        print("splitting dataframe for training & testing...")
        train_df, test_df = self.split_train_test(df)
        
        df_dict = {'train':train_df, 'test':test_df}
        
        def parallel_snapshot_graphs(df, period):
            df_snap = df[df[self.time_index_col]<=period]
            # ensure only required context is taken
            df_snap = df_snap.groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True)
            #print(df_snap.shape)
            snapshot_graph = self.create_snapshot_graph(df_snap, period)
            return snapshot_graph
        
        # for each split create graph dataset iterator
        print("gather snapshot graphs...")
        datasets = {}
        for df_type, df in df_dict.items():
            # snapshot start period: time.min() + max_history + fh, end_period:
            
            snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)[self.window_len-1:]
            
            # restrict samples for very large datasets based on interleaving
            if (self.interleave > 1) and (df_type == 'train'):
                snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]
            
            print("picking {} samples for {}".format(len(snap_periods_list), df_type))
            #print(snap_periods_list)
            
            snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)

            # Create a dataset iterator
            dataset = DataLoader(snapshot_list, batch_size=self.batch, shuffle=self.shuffle) # Load full graph for each timestep
            
            # append
            datasets[df_type] = dataset
        
        train_dataset, test_dataset = datasets.get('train'), datasets.get('test')

        return train_dataset, test_dataset
    
    def create_infer_dataset(self, df, infer_till):
        
        self.infer_till = infer_till
        
        # preprocess
        #print("preprocessing dataframe...")
        df = self.preprocess(df)
        
        # pad dataframe if required (will return df unchanged if not)
        #print("padding dataframe...")
        df = self.pad_dataframe(df)
        
        # split into train,test,infer
        infer_df = self.split_infer(df)
        
        #infer_df = self.pad_dataframe(infer_df)
        df_dict = {'infer':infer_df}
        
        # for each split create graph dataset iterator
        datasets = {}
        for df_type, df in df_dict.items():
            # snapshot start period: time.min() + max_history + fh, end_period:
            
            snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)[-1]
            #print("inference snapshot period: ",snap_periods_list)
            
            # create individual snapshot graphs
            snapshot_list = []
            for period in [snap_periods_list]:
                df_snap = df[df[self.time_index_col]<=period]
                # ensure only required context is taken
                df_snap = df_snap.groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True)
                #print(df_snap.shape)
                snapshot_graph = self.create_snapshot_graph(df_snap, period)
                snapshot_list.append(snapshot_graph)

            # Create a dataset iterator
            dataset = DataLoader(snapshot_list, batch_size=1, shuffle=False) 
            
            # append
            datasets[df_type] = dataset
        
        infer_dataset = datasets.get('infer')

        return infer_dataset
    
    
    def split_train_test(self, data):
        
        # adjust train_till
        lead_forecast_delta = self.max_leads - self.fh
        
        all_ts = sorted(data[self.time_index_col].unique().tolist())
        current_train_till_idx = all_ts.index(self.train_till)
        new_train_till_idx = int(current_train_till_idx + lead_forecast_delta)
        self.adj_train_till = all_ts[new_train_till_idx]
        
        # adjust test_till
        current_test_till_idx = all_ts.index(self.test_till)
        new_test_till_idx = int(current_test_till_idx + lead_forecast_delta)
        self.adj_test_till = all_ts[new_test_till_idx]
        
        train_data = data[data[self.time_index_col]<=self.adj_train_till].reset_index(drop=True)
        
        # test
        test_len = int(data[(data[self.time_index_col]>self.train_till) & (data[self.time_index_col]<=self.adj_test_till)].groupby(self.id_col)[self.target_col].count().max())
        test_len = test_len + (self.window_len - self.max_leads)
        test_data = data[data[self.time_index_col]<=self.adj_test_till].groupby(self.id_col).apply(lambda x: x[-test_len:]).reset_index(drop=True)
        
        return train_data, test_data
    
    def split_infer(self, data):
        
        # adjust infer_till
        lead_forecast_delta = self.max_leads - self.fh
        all_ts = sorted(data[self.time_index_col].unique().tolist())
        
        current_infer_till_idx = all_ts.index(self.infer_till)
        new_infer_till_idx = int(current_infer_till_idx + lead_forecast_delta)
        self.adj_infer_till = all_ts[new_infer_till_idx]
        
        infer_len = self.window_len
        infer_data = data[data[self.time_index_col]<=self.adj_infer_till].groupby(self.id_col).apply(lambda x: x[-infer_len:]).reset_index(drop=True)
        
        return infer_data

    def get_metadata(self, dataset):
        
        batch = next(iter(dataset))
        
        return batch.metadata()
    
    def show_batch_statistics(self, dataset):
        
        batch = next(iter(dataset))
        statistics = {}
        
        statistics['nodetypes'] = list(batch.x_dict.keys())
        statistics['edgetypes'] = list(batch.edge_index_dict.keys())
        statistics['num_nodes'] = batch.num_nodes
        statistics['num_target_nodes'] = int(batch[self.target_col].num_nodes/self.max_history)
        statistics['num_edges'] = batch.num_edges
        statistics['node_feature_dims'] = batch.num_node_features
        statistics['max_lags'] = batch[self.target_col].num_node_features
        statistics['target_feature_dims'] = batch[self.target_col].y.shape
        statistics['adjusted_train_till'] = self.adj_train_till
        statistics['adjusted_test_till'] = self.adj_test_till
        
        return statistics
      
    def process_output(self, df, model_output):
        
        if self.include_rolling_features:
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.rolling_stat_cols))
        if not self.categorical_onehot_encoding:
            self.temporal_known_num_col_list = list(set(self.temporal_known_num_col_list) - set(self.label_encoded_col_list))
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.label_encoded_col_list))
            
        # preprocess
        print("preprocessing dataframe...")
        df = self.preprocess(df)
        
        # pad dataframe if required (will return df unchanged if not)
        print("padding dataframe...")
        df = self.pad_dataframe(df)
        
        # get infer df
        infer_df = self.split_infer(df)
        
        # groupby target node ids
        if (self.max_leads-self.fh) >= 1:
            infer_df = infer_df.groupby(self.id_col, sort=False).apply(lambda x: x[-self.max_leads:-(self.max_leads-self.fh)]).reset_index(drop=True)
            print(infer_df[self.time_index_col].unique().tolist())
        else:
            infer_df = infer_df.groupby(self.id_col, sort=False).apply(lambda x: x[-self.max_leads:]).reset_index(drop=True)
            print(infer_df[self.time_index_col].unique().tolist())
            
        # infer_df.shape[0] == model_output.shape[0]
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            scaler_cols = ['scaler']
        else:
            scaler_cols = ['scaler_mu','scaler_std']
        
        infer_df = infer_df[[self.id_col, self.target_col, self.time_index_col] + self.static_cat_col_list + self.global_context_col_list + scaler_cols]
        
        model_output = model_output.reshape(-1,1)
        output = pd.DataFrame(data=model_output, columns=['forecast'])
        
        # merge forecasts with infer df
        output = pd.concat([infer_df, output], axis=1)    
       
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            output['forecast'] = output['forecast']*output['scaler']
            output[self.target_col] = output[self.target_col]*output['scaler']
        else:
            output['forecast'] = output['forecast']*output['scaler_std'] + output['scaler_mu']
            output[self.target_col] = output[self.target_col]*output['scaler_std'] + output['scaler_mu']
        
        return output
        
    def update_dataframe(self, df, output):
        
        # merge output & base_df
        reduced_output_df = output[[self.id_col, self.time_index_col, 'forecast']]
        df_updated = df.merge(reduced_output_df, on=[self.id_col, self.time_index_col], how='left')
        
        # update target for current ts with forecasts
        df_updated[self.target_col] = np.where(df_updated['forecast'].isnull(), df_updated[self.target_col], df_updated['forecast'])
        
        # drop forecast column
        df_updated = df_updated.drop(columns=['forecast'])
        
        # also, drop rolling stats cols if selected
        if self.include_rolling_features:
            # reset
            self.temporal_unknown_num_col_list = []
            
        return df_updated
    
    def build_dataset(self, df):
        # build graph datasets for train/test
        self.train_dataset, self.test_dataset = self.create_train_test_dataset(df)

    def build_infer_dataset(self, df, infer_till):
        # build graph datasets for infer
        try:
            del self.infer_dataset
            gc.collect()
        except:
            pass
        self.infer_dataset = self.create_infer_dataset(df=df, infer_till=infer_till)

    def build(self,
              model_type = "SAGE", 
              model_option = "LSTMATTENTION", 
              model_dim = 128,
              num_layers = 1,
              lstm_layers = 1,
              attention_heads = 1,
              forecast_quantiles = [0.5, 0.55, 0.60, 0.65, 0.70],
              dropout = 0.0,
              residual_conn_type = 'concat',
              aggr = 'mean',
              use_linear_pretransform = True,
              apply_norm_layers = True,
              gnn_skip_connection = False,
              use_dirgnn = True,
              device = 'cpu'):
        
        # key metadata for model def
        self.metadata = self.get_metadata(self.train_dataset)
        self.n_prediction_nodes = self.show_batch_statistics(self.train_dataset).get('num_target_nodes')
        self.forecast_quantiles = forecast_quantiles
        sample_batch = next(iter(self.train_dataset))
        
        # target device to train on ['cuda','cpu']
        self.device = torch.device(device)
        
        # build model
        self.model = STGNN(model_type = model_type,
                           model_option = model_option,
                           hidden_channels = model_dim, 
                           out_channels = self.fh, 
                           metadata = self.metadata, 
                           n_nodes = self.n_prediction_nodes, 
                           target_node = self.target_col,
                           context_nodes = self.context_cols,
                           seq_len = self.max_history,
                           device = self.device,
                           heads = attention_heads,
                           n_quantiles = max(1, len(self.forecast_quantiles)),
                           num_layers = num_layers,
                           lstm_layers = lstm_layers,
                           alpha = 0.5, 
                           dropout = dropout,
                           residual_conn_type = residual_conn_type,
                           positive_output = False,
                           aggr = aggr,
                           use_linear_pretransform = use_linear_pretransform,
                           apply_norm_layers = apply_norm_layers,
                           skip_connection = gnn_skip_connection,
                           use_dirgnn = use_dirgnn)
        
        # init model
        self.model = self.model.to(self.device)
        
        # Lazy init.
        with torch.no_grad():
            sample_batch = sample_batch.to(self.device)
            out = self.model(sample_batch.x_dict, sample_batch.edge_index_dict)
            
        # parameters count
        try:
            pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("total model params: ", pytorch_total_params)
        except:
            pytorch_total_params = sum([0 if isinstance(p, torch.nn.parameter.UninitializedParameter) else p.numel() for p in self.model.parameters()])
            print("total model params: ", pytorch_total_params)
            
            
    def train(self, 
              lr, 
              min_epochs, 
              max_epochs, 
              patience, 
              min_delta, 
              model_prefix,
              loss_type = 'Quantile',
              delta = 1.0,
              use_lr_scheduler=True, 
              scheduler_params={'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001}):
        
        self.loss_type = loss_type
        
        if self.loss_type == 'Quantile':
            loss_fn = QuantileLoss(quantiles=self.forecast_quantiles)
        elif self.loss_type == 'Huber':
            loss_fn = torch.nn.HuberLoss(reduction='none', delta=delta)
        else:
            loss_fn = RMSE()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   mode='min', 
                                                                   factor=scheduler_params['factor'], 
                                                                   patience=scheduler_params['patience'], 
                                                                   threshold=scheduler_params['threshold'], 
                                                                   threshold_mode='rel', 
                                                                   cooldown=0, 
                                                                   min_lr=scheduler_params['min_lr'], 
                                                                   eps=1e-08, 
                                                                   verbose=False)
        
        # init training data structures & vars
        model_list = []
        self.best_model = None
        time_since_improvement = 0
        train_loss_hist = []
        val_loss_hist = []
        
        def train_fn():
            self.model.train(True)
            total_examples = 0 
            total_loss = 0
            for i, batch in enumerate(self.train_dataset):
                optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_size = batch.num_graphs
                out = self.model(batch.x_dict, batch.edge_index_dict)
                
                # compute loss masking out N/A targets -- last snapshot
                if self.loss_type == 'Quantile':
                    try:
                        loss = loss_fn.loss(out[-self.n_prediction_nodes:, :, :],
                                            batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    except:
                        loss = loss_fn.loss(torch.unsqueeze(out[-self.n_prediction_nodes:, :], dim=1),
                                            batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    mask = torch.unsqueeze(batch[self.target_col].y_mask[-self.n_prediction_nodes:, :], dim=2)
                elif self.loss_type == 'Huber':
                    try:
                        loss = loss_fn(out[-self.n_prediction_nodes:, -1, :],
                                       batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    except:
                        loss = loss_fn(out[-self.n_prediction_nodes:, :],
                                       batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    mask = batch[self.target_col].y_mask[-self.n_prediction_nodes:, :]
                else:
                    try:
                        loss = loss_fn.loss(out[-self.n_prediction_nodes:, -1, :],
                                            batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    except:
                        loss = loss_fn.loss(out[-self.n_prediction_nodes:, :],
                                            batch[self.target_col].y[-self.n_prediction_nodes:, :])
                    mask = batch[self.target_col].y_mask[-self.n_prediction_nodes:, :]
                
                """
                # compute loss masking out N/A targets -- all snapshot
                #print(out.shape, batch[self.target_col].y.shape)
                # snapshot weights
                snap_wts = np.concatenate([[i+1]*self.n_prediction_nodes for i in range(self.max_history)], axis=0).reshape(-1,1)
                snap_wts = torch.tensor(snap_wts, dtype=torch.float).to(self.device)
                
                target_batches = []
                mask_batches = []
                snap_wts_batches = []
                
                for i in range(self.max_history):
                    batch_step = batch[self.target_col].y[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                    batch_step = torch.unsqueeze(batch_step, dim=1)
                    target_batches.append(batch_step)
                    
                    mask_batch_step = batch[self.target_col].y_mask[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                    mask_batch_step = torch.unsqueeze(mask_batch_step, dim=1)
                    mask_batches.append(mask_batch_step)
                    
                    snap_wts_batch_step = snap_wts[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                    snap_wts_batch_step = torch.unsqueeze(snap_wts_batch_step, dim=1)
                    snap_wts_batches.append(snap_wts_batch_step)

                target = torch.squeeze(torch.concat(target_batches, dim=1), dim=2)
                mask = torch.concat(mask_batches, dim=1)
                snapwt = torch.concat(snap_wts_batches, dim=1)
                loss = loss_fn.loss(out, target)
                wts = mask
                #print(loss.shape)
                
                """
        
                # apply mask & wts
                loss = torch.mean(loss*mask)
                loss.backward()
                optimizer.step()
                total_examples += batch_size
                total_loss += float(loss)
                
            return total_loss / total_examples
        
        def test_fn():
            self.model.train(False) # original line
            total_examples = 0 
            total_loss = 0
            with torch.no_grad(): 
                for i, batch in enumerate(self.test_dataset):
                    batch_size = batch.num_graphs
                    batch = batch.to(self.device)
                    out = self.model(batch.x_dict, batch.edge_index_dict)
                    
                    # compute loss masking out N/A targets -- last snapshot
                    if self.loss_type == 'Quantile':
                        try:
                            loss = loss_fn.loss(out[-self.n_prediction_nodes:, :, :],
                                                batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        except:
                            loss = loss_fn.loss(torch.unsqueeze(out[-self.n_prediction_nodes:, :], dim=1),
                                                batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        mask = torch.unsqueeze(batch[self.target_col].y_mask[-self.n_prediction_nodes:, :], dim=2)
                    elif self.loss_type == 'Huber':
                        try:
                            loss = loss_fn(out[-self.n_prediction_nodes:, -1, :],
                                           batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        except:
                            loss = loss_fn(out[-self.n_prediction_nodes:, :],
                                           batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        mask = batch[self.target_col].y_mask[-self.n_prediction_nodes:, :]
                    else:
                        try:
                            loss = loss_fn.loss(out[-self.n_prediction_nodes:, -1, :],
                                                batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        except:
                            loss = loss_fn.loss(out[-self.n_prediction_nodes:, :],
                                                batch[self.target_col].y[-self.n_prediction_nodes:, :])
                        mask = batch[self.target_col].y_mask[-self.n_prediction_nodes:, :]
                    
                    """
                    # compute loss masking out N/A targets -- all snapshot
                    # calculate snapshot weights
                    snap_wts = np.concatenate([[i+1]*self.n_prediction_nodes for i in range(self.max_history)], axis=0).reshape(-1,1)
                    snap_wts = torch.tensor(snap_wts, dtype=torch.float).to(self.device)

                    target_batches = []
                    mask_batches = []
                    snap_wts_batches = []

                    for i in range(self.max_history):
                        batch_step = batch[self.target_col].y[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                        batch_step = torch.unsqueeze(batch_step, dim=1)
                        target_batches.append(batch_step)

                        mask_batch_step = batch[self.target_col].y_mask[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                        mask_batch_step = torch.unsqueeze(mask_batch_step, dim=1)
                        mask_batches.append(mask_batch_step)

                        snap_wts_batch_step = snap_wts[int(i*self.n_prediction_nodes):int((i+1)*self.n_prediction_nodes)]
                        snap_wts_batch_step = torch.unsqueeze(snap_wts_batch_step, dim=1)
                        snap_wts_batches.append(snap_wts_batch_step)

                    target = torch.squeeze(torch.concat(target_batches, dim=1), dim=2)
                    mask = torch.concat(mask_batches, dim=1)
                    snapwt = torch.concat(snap_wts_batches, dim=1)
                    loss = loss_fn.loss(out, target)
                    wts = mask
                    
                    """
                    
                    # apply mask & wts
                    loss = torch.mean(loss*mask) 
                    total_examples += batch_size
                    total_loss += float(loss)
                    
            return total_loss / total_examples
        
        for epoch in range(max_epochs):
            
            loss = train_fn()
            val_loss = test_fn()
            
            print('EPOCH {}: Train loss: {}, Val loss: {}'.format(epoch, loss, val_loss))
            
            if use_lr_scheduler:
                scheduler.step(val_loss)

            train_loss_hist.append(loss)
            val_loss_hist.append(val_loss)

            model_path = model_prefix + '_' + str(epoch) 
            model_list.append(model_path)
            
            # compare loss
            if epoch == 0:
                prev_min_loss = np.min(val_loss_hist)
            else:
                prev_min_loss = np.min(val_loss_hist[:-1])

            current_min_loss = np.min(val_loss_hist)
            delta = current_min_loss - prev_min_loss

            save_condition = ((val_loss_hist[epoch] == np.min(val_loss_hist)) and (-delta > min_delta)) or (epoch == 0)

            print("Improvement delta (min_delta {}):  {}".format(min_delta, delta))

            # track & save best model
            if save_condition:
                self.best_model = model_path
                # save model
                torch.save(self.model.state_dict(), model_path)
                # reset time_since_improvement
                time_since_improvement = 0
            else:
                time_since_improvement += 1

            # remove older models
            if len(model_list)>patience:
                for m in model_list[:-patience]:
                    if m != self.best_model:
                        try:
                            shutil.rmtree(m)
                        except:
                            pass

            if ((time_since_improvement > patience) and (epoch > min_epochs)) or (epoch == max_epochs - 1):
                print("Terminating Training. Best Model: {}".format(self.best_model))
                break

    def change_device(self, device='cpu'):
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(self.best_model, map_location=self.device))

    def disable_cuda_backend(self,):
        self.change_device(device="cuda")
        torch.backends.cudnn.enabled = False
    def infer(self, df, infer_start, infer_end, select_quantile, compute_mape=False):
        
        base_df = df.copy()
        
        # get list of infer periods
        infer_periods = sorted(base_df[(base_df[self.time_index_col]>=infer_start) & (base_df[self.time_index_col]<=infer_end)][self.time_index_col].unique().tolist())
        
        # print model used for inference
        print("running inference using best saved model: ", self.best_model)
        
        forecast_df = pd.DataFrame() 
        
        # infer fn
        def infer_fn(model, model_path, infer_dataset):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            output = []
            with torch.no_grad(): 
                for i, batch in enumerate(infer_dataset):
                    batch = batch.to(self.device)
                    out = model(batch.x_dict, batch.edge_index_dict)
                    output.append(out)
            return output

        for i,t in enumerate(infer_periods):
            
            print("forecasting period {} at lag {}".format(t, i))
            
            # reset rolling stats columns -- will be recalculated for each period & undo labelencoding & scaling
            if self.include_rolling_features:
                self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.rolling_stat_cols))
            
            if not self.categorical_onehot_encoding:
                self.temporal_known_num_col_list = list(set(self.temporal_known_num_col_list) - set(self.label_encoded_col_list))
                self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.label_encoded_col_list))
        
            # infer dataset creation 
            infer_dataset = self.create_infer_dataset(base_df, infer_till=t)
            output = infer_fn(self.model, self.best_model, infer_dataset)
            
            # select output quantile
            output_arr = output[0]
            output_arr = output_arr.cpu().numpy()
            
            # quantile selection
            min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

            if self.loss_type == 'Quantile':
                assert select_quantile >= min_qtile and select_quantile <= max_qtile, "selected quantile out of bounds!"
                try:
                    q_index = self.forecast_quantiles(select_quantile)
                    output_arr = output_arr[-self.n_prediction_nodes:,:,q_index] 
                except:
                    q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                    q_lower = int(q_upper - 1)
                    q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower] )/(self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                    q_lower_weight = 1 - q_upper_weight
                    output_arr = q_upper_weight*output_arr[-self.n_prediction_nodes:,:,q_upper] + q_lower_weight*output_arr[-self.n_prediction_nodes:,:,q_lower]
            else:
                try:
                    output_arr = output_arr[:, :, 0]
                except:
                    pass
                
            # show current o/p
            scaled_output = self.process_output(base_df, output_arr)
            
            # compute mape
            if compute_mape:
                scaled_output['forecast_sum'] = scaled_output.groupby(self.id_col)['forecast'].transform(lambda x: x.sum())
                scaled_output['forecast_sum'] = np.where(scaled_output['forecast_sum']<0, 0, scaled_output['forecast_sum'])
                scaled_output['abs_error'] = abs(scaled_output[self.target_col] - scaled_output['forecast'])
                print(scaled_output.groupby(self.time_index_col)[['forecast',self.target_col,'abs_error']].sum())
                
            # append forecast
            forecast_df = pd.concat([forecast_df, scaled_output], axis=0)

            # update df
            base_df = self.update_dataframe(base_df, scaled_output)
        
        return forecast_df
    
    
    def infer_oneshot(self, df, select_quantile, compute_mape=False):
        
        base_df = df.copy()
        
        # get list of infer periods
        infer_periods = sorted(base_df[(base_df[self.time_index_col]>=infer_start) & (base_df[self.time_index_col]<=infer_end)][self.time_index_col].unique().tolist())
        
        # print model used for inference
        print("running inference using best saved model: ", self.best_model)
        
        # infer fn
        def infer_fn(model, model_path, infer_dataset):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.train(False)
            output = []
            with torch.no_grad(): 
                for i, batch in enumerate(infer_dataset):
                    batch = batch.to(self.device)
                    out = model(batch.x_dict, batch.edge_index_dict)
                    output.append(out)
            return output

        print("forecasting for periods from {} to {}".format(infer_periods[0], infer_periods[-1]))
            
        # reset rolling stats columns -- will be recalculated for each period & undo labelencoding & scaling
        if self.include_rolling_features:
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.rolling_stat_cols))
            
        if not self.categorical_onehot_encoding:
            self.temporal_known_num_col_list = list(set(self.temporal_known_num_col_list) - set(self.label_encoded_col_list))
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.label_encoded_col_list))
        
        # infer dataset creation 
        infer_dataset = self.create_infer_dataset(base_df, infer_till=infer_periods[-1])
        output = infer_fn(self.model, self.best_model, infer_dataset)
            
        # select output quantile
        output_arr = output[0]
        output_arr = output_arr.cpu().numpy()
            
        # quantile selection
        min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

        if self.loss_type == 'Quantile':
            assert select_quantile >= min_qtile and select_quantile <= max_qtile, "selected quantile out of bounds!"
            try:
                q_index = self.forecast_quantiles(select_quantile)
                output_arr = output_arr[-self.n_prediction_nodes:,:,q_index]
            except:
                q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                q_lower = int(q_upper - 1)
                q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower] )/(self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                q_lower_weight = 1 - q_upper_weight
                output_arr = q_upper_weight*output_arr[-self.n_prediction_nodes:,:,q_upper] + q_lower_weight*output_arr[-self.n_prediction_nodes:,:,q_lower]
        else:
            try:
                output_arr = output_arr[:, :, 0]
            except:
                pass

        # show current o/p
        scaled_output = self.process_output(base_df, output_arr)
            
        # compute mape
        if compute_mape:
            scaled_output['abs_error'] = abs(scaled_output[self.target_col] - scaled_output['forecast'])
            print(scaled_output.groupby(self.time_index_col)[['forecast',self.target_col,'abs_error']].sum())
                
        return scaled_output

        
                

