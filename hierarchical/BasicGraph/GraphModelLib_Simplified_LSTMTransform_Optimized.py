#!/usr/bin/env python
# coding: utf-8
import random

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
from ast import literal_eval
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import shutil
import sys
import time

os = sys.platform

if os == 'linux':
    backend = 'loky'
    timeout = 3600
else:
    backend = 'loky'
    timeout = 3600

# set default dtype to float32
torch.set_default_dtype(torch.float32)

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
            self.transform_layers = torch.nn.ModuleList()
            for _ in range(1):
                transformed_feat_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    #if node_type == self.target_node_type:
                    transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                     hidden_size=hidden_channels,
                                                                     num_layers=1,
                                                                     batch_first=True)
                    #else:
                    #    transformed_feat_dict[node_type] = Linear(in_channels=-1, out_channels=hidden_channels)
                        
                self.transform_layers.append(transformed_feat_dict)

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
                        conv_dict[e] = SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels)
                elif (e[0] in self.context_node_type) or (e[2] in self.context_node_type):
                    # global context nodes
                    conv_dict[e] = SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels)
                else:
                    if i == 0:
                        conv_dict[e] = SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels)
                    else:
                        # layers after first layer operate only on demand nodes & not covariates
                        pass
                    
            conv = HeteroConv(conv_dict, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, hidden_channels)
        
    def apply_linear_layer(self, x_dict, tfr_dict):
        transformed_x_dict = {}
        for node_type, x in x_dict.items():
            #if node_type == self.target_node_type:
            o, _ = tfr_dict[node_type](torch.unsqueeze(x, dim=2)) # lstm input is 3 -d (N,L,1)
            transformed_x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
            #else:
            #    transformed_x_dict[node_type] = tfr_dict[node_type](x)
                
        return transformed_x_dict

    def forward(self, x_dict, edge_index_dict):
        
        if self.use_linear_pretransform:
            # lstm transform node features
            for tfr_dict in self.transform_layers:
                x_dict = self.apply_linear_layer(x_dict, tfr_dict)
        
        if self.skip_connection:        
            res_x_dict = x_dict
           
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
            self.transform_layers = torch.nn.ModuleList()
            for _ in range(1):
                transformed_feat_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    # if node_type == self.target_node_type:
                    transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                     hidden_size=hidden_channels,
                                                                     num_layers=1,
                                                                     batch_first=True)
                    # else:
                    #    transformed_feat_dict[node_type] = Linear(in_channels=-1, out_channels=hidden_channels)
                self.transform_layers.append(transformed_feat_dict)

        """
        if self.use_linear_pretransform:  
            self.linear_layers = torch.nn.ModuleList()
            lin_dict = HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=node_types)
            self.linear_layers.append(lin_dict)
        """

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

    def apply_linear_layer(self, x_dict, tfr_dict):
        transformed_x_dict = {}
        for node_type, x in x_dict.items():
            # if node_type == self.target_node_type:
            o, _ = tfr_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3 -d (N,L,1)
            transformed_x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
            # else:
            #    transformed_x_dict[node_type] = tfr_dict[node_type](x)

        return transformed_x_dict

    def forward(self, x_dict, edge_index_dict):

        if self.use_linear_pretransform:
            # lstm transform node features
            for tfr_dict in self.transform_layers:
                x_dict = self.apply_linear_layer(x_dict, tfr_dict)

        """
        if self.use_linear_pretransform:  
            # linear transform node features
            for lin_dict in self.linear_layers:
                x_dict = lin_dict(x_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
        """

        if self.skip_connection:        
            res_x_dict = x_dict 

        # apply dir sage layer to transformed dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            if self.skip_connection:
                res_x_dict = {key: res_x_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x) in zip(x_dict.items(), res_x_dict.items()) if key == res_key}
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
                 heads,
                 metadata, 
                 target_node,
                 context_nodes,
                 device,
                 n_quantiles=1, 
                 num_layers=1,
                 alpha=0.5, 
                 dropout=0.0,
                 residual_conn_type='concat',
                 loss_type='Quantile', 
                 positive_output=False,
                 aggr='mean',
                 use_linear_pretransform=True,
                 apply_norm_layers=True,
                 skip_connection=True,
                 use_dirgnn=True):
        
        """
        model options:
        
        model_type: ['SAGE','GCN','GAT','GATV2']
        model_options: ['BASIC']
        loss_type: ['Point','Quantile']
        positive_output: True/False
        residual_conn_type; ['add','concat']
        
        """
        
        super(STGNN, self).__init__()
        
        self.model_type = str.upper(model_type)
        self.model_option = str.upper(model_option)
        
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.n_pred = out_channels
        self.dropout = dropout
        self.target_node = target_node
        self.context_nodes = context_nodes
        self.device = device
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.alpha = alpha
        self.dropout = dropout
        self.pos_out = positive_output
        self.n_quantiles = n_quantiles
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.apply_norm_layers = apply_norm_layers
        self.use_dirgnn = use_dirgnn
        self.skip_connection = skip_connection
        

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
        
        else:
            raise "Invalid model_option. model_option: [BASIC]"
            
    def forward(self, x_dict, edge_index_dict):

        # get keybom
        keybom = x_dict['keybom']

        # get key_aggregation_status
        key_agg_status = x_dict['key_aggregation_status']
        agg_indices = (key_agg_status == 1).nonzero(as_tuple=True)[0].tolist()

        # del keybom from x_dict
        del x_dict['keybom']
        del x_dict['key_aggregation_status']

        # gnn layer
        x = self.gnn_layer(x_dict, edge_index_dict)
        x = x.relu()
        
        if self.apply_norm_layers:
            x = self.layer_norm1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # output preds now without temporal attention
        
        if self.model_option == "BASIC":
            
            # final projection layer
            out = self.project_lin(x)
            
            if self.pos_out:
                out = F.softplus(out)

            if self.n_quantiles > 1:
                out = torch.reshape(out, (-1, self.n_pred, self.n_quantiles))
            else:
                out = torch.reshape(out, (-1, self.n_pred))

            # constrain the higher level key o/ps to be the sum of their constituents
            for i in agg_indices:
                out[i] = torch.index_select(out, 0, keybom[i][keybom[i] != -1]).sum(dim=0)

        return out
    

# #### Graph Data Generator

class graphmodel():
    def __init__(self, 
                 col_dict, 
                 max_lags,
                 max_leads,
                 train_till,
                 test_till,
                 fh = 1,
                 batch = 1,
                 grad_accum = False,
                 accum_iter = 1,
                 scaling_method = 'mean_scaling',
                 iqr_high = 0.75,
                 iqr_low = 0.25,
                 categorical_onehot_encoding = True,
                 directed_graph = True,
                 shuffle = True,
                 interleave = 1,
                 recency_weights = False,
                 recency_alpha = 0,
                 PARALLEL_DATA_JOBS = 4, 
                 PARALLEL_DATA_JOBS_BATCHSIZE = 128):
        """
        col_dict: dictionary of various column groups {id_col:'',
                                                       key_combinations:[(),(),...],
                                                       key_combination_weights:{key_combination_1: wt, key_combination_2: wt,...}
                                                       lowest_key_combination: (),
                                                       highest_key_combination: (),
                                                       target_col: '',
                                                       time_index_col: '',
                                                       global_context_col_list: [],
                                                       static_cat_col_list: [],
                                                       subgraph_samples_col:None,
                                                       subgraph_sample_size:100,
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
        self.max_history = int(1)
        self.max_lags = int(max_lags) if (max_lags is not None) and (max_lags>0) else 1
        self.max_leads = int(max_leads) if (max_leads is not None) and (max_leads>0) else 1

        assert self.max_leads >= self.fh, "max_leads must be >= fh"
        
        # adjust train_till/test_till for delta|max_leads - fh| in split_* methods
        self.train_till = train_till
        self.test_till = test_till
        
        self.batch = batch
        self.grad_accum = grad_accum
        self.accum_iter = accum_iter
        self.scaling_method = scaling_method
        self.iqr_high = iqr_high
        self.iqr_low = iqr_low
        self.categorical_onehot_encoding = categorical_onehot_encoding
        self.directed_graph = directed_graph
        self.shuffle = shuffle
        self.interleave = interleave
        self.recency_weights = recency_weights
        self.recency_alpha = recency_alpha
        self.PARALLEL_DATA_JOBS = PARALLEL_DATA_JOBS
        self.PARALLEL_DATA_JOBS_BATCHSIZE = PARALLEL_DATA_JOBS_BATCHSIZE
        
        self.pad_constant = 0  # if self.scaling_method == 'mean_scaling' else -1
       
        # extract columnsets from col_dict
        # hierarchy specific keys
        self.id_col = self.col_dict.get('id_col')
        self.key_combinations = self.col_dict.get('key_combinations')
        self.key_combination_weights = self.col_dict.get('key_combination_weights', None)
        self.lowest_key_combination = self.col_dict.get('lowest_key_combination')
        self.highest_key_combination = self.col_dict.get('highest_key_combination')
        self.new_key_cols = []
        self.key_levels_dict = {}
        self.key_levels_weight_dict = {}
        self.covar_key_level = None
        self.key_targets_dict = {}

        self.target_col = self.col_dict.get('target_col', None)
        self.time_index_col = self.col_dict.get('time_index_col', None)
        self.datetime_format = self.col_dict.get('datetime_format', None)
        self.strata_col_list = self.col_dict.get('strata_col_list', [])
        self.sort_col_list = self.col_dict.get('sort_col_list', [self.id_col, self.time_index_col])
        self.wt_col = self.col_dict.get('wt_col', None)
        self.global_context_col_list = self.col_dict.get('global_context_col_list', [])
        self.static_cat_col_list = self.col_dict.get('static_cat_col_list', [])
        self.subgraph_sample_col = self.col_dict.get('subgraph_sample_col', None)
        self.subgraph_sample_size = self.col_dict.get('subgraph_sample_size', 100)
        self.temporal_known_num_col_list = self.col_dict.get('temporal_known_num_col_list', [])
        self.temporal_unknown_num_col_list = self.col_dict.get('temporal_unknown_num_col_list', [])
        self.temporal_known_cat_col_list = self.col_dict.get('temporal_known_cat_col_list', [])
        self.temporal_unknown_cat_col_list = self.col_dict.get('temporal_unknown_cat_col_list', [])

        if (self.id_col is None) or (self.target_col is None) or (self.time_index_col is None) or (self.key_combinations is None) or (self.lowest_key_combination is None):
            raise ValueError("Id Column, Target Column or Index Column not specified!")

        # check for tuples in key_combinations
        for k in self.key_combinations:
            res = type(k) is tuple
            if not res:
                raise ValueError("non-tuple found in key_combinations list!")

        # check for non-tuple in lowest_key_combination
        if (type(self.lowest_key_combination) is not tuple) or (type(self.highest_key_combination) is not tuple):
            raise ValueError("non-tuple found for lowest_key_combination or highest_key_combination!")

        # full columnset for train/test/infer
        self.col_list = [self.id_col] + [self.target_col] + \
                         self.static_cat_col_list + self.global_context_col_list + \
                         self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                         self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.cat_col_list = self.global_context_col_list + self.static_cat_col_list + \
                            self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.temporal_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + \
                                 self.temporal_known_num_col_list + self.temporal_unknown_num_col_list

    def create_new_keys(self, df):
        for i, k in enumerate(self.key_combinations):
            key = "key_" + "_".join(k)
            self.new_key_cols.append(key)
            df[key] = df[list(k)].astype(str).apply(lambda x: "_".join(x), axis=1)
            self.key_levels_dict[key] = list(k)  # (",".join(k))
            if k == self.lowest_key_combination:
                self.covar_key_level = key  # ",".join(k)

        if self.key_combination_weights is not None:
            for k, v in self.key_combination_weights.items():
                key = "key_" + "_".join(k)
                self.key_levels_weight_dict[key] = v
        else:
            for key, _ in self.key_levels_dict.items():
                self.key_levels_weight_dict[key] = 1

        print("created new key cols: ", self.new_key_cols)
        print("created new key to subkeys mapping: ", self.key_levels_dict)
        print("covariates applied at this key level: ", self.covar_key_level)
        return df

    def create_new_targets(self, df):
        for key in self.new_key_cols:
            df[f'{key}_target'] = df.groupby([key, self.time_index_col])[self.target_col].transform(lambda x: x.sum())
            self.key_targets_dict[key] = f'{key}_target'
        return df

    def get_keybom(self, df):
        """
        For every key at every key_level, obtain a list of constituent keys
        """
        keybom_list = []
        for key in self.new_key_cols:
            df_key_map = df.groupby([key, self.time_index_col])[self.covar_key_level].apply(lambda x: x.unique().tolist()).rename('key_list').reset_index().rename(columns={key: self.id_col})
            keybom_list.append(df_key_map)
        df_keybom = pd.concat(keybom_list, axis=0)
        df_keybom = df_keybom.reset_index(drop=True)

        return df_keybom

    def stack_key_level_dataframes(self, df, df_keybom):
        df_stack_list = []
        for (k, v), (k2, v2) in zip(self.key_levels_dict.items(), self.key_targets_dict.items()):
            if k == k2:
                if k == self.covar_key_level:
                    df_temp = df[[k, v2, self.time_index_col] + v + self.global_context_col_list + self.static_cat_col_list + self.temporal_col_list]
                    df_temp = df_temp.drop_duplicates()
                else:
                    df_temp = df[[k, v2, self.time_index_col] + v + self.global_context_col_list + self.static_cat_col_list]
                    df_temp = df_temp.drop_duplicates(subset=[k, v2, self.time_index_col] + v)
                df_temp[self.id_col] = df[k]
                df_temp[self.target_col] = df[v2]
                df_temp["key_level"] = k
                df_temp = df_temp.merge(df_keybom, on=[self.id_col, self.time_index_col], how='inner')
                df_temp = df_temp.drop(columns=[k, v2])
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()].reset_index(drop=True)
                df_stack_list.append(df_temp)

        # replace df with stacked dataframe
        df = pd.concat(df_stack_list, axis=0, ignore_index=True)
        # Add 'Key_Weight' col
        df['Key_Weight'] = df['key_level'].map(self.key_levels_weight_dict)
        # new col list
        self.col_list = df.columns.tolist()
        return df

    def scale_target(self, df):
        """
        Scale using scalers for the highest key combination in the hierarchy
        """
        if self.scaling_method == 'mean_scaling':
            highest_key_cols = list(self.highest_key_combination)
            df['scaler'] = df[df[self.time_index_col] <= self.train_till].groupby(highest_key_cols)[self.target_col].transform(lambda x: np.maximum(x.mean(), 1.0))
            df['scaler'] = df.groupby(highest_key_cols)['scaler'].transform(lambda x: x.ffill().bfill())
            df[self.target_col] = df[self.target_col]/df['scaler']

        elif self.scaling_method == 'no_scaling':
            df['scaler'] = 1.0
            df[self.target_col] = df[self.target_col] / df['scaler']

        return df

    def scale_covariates(self, df):
        """
        Individually scale each 'id' & concatenate them all in one dataframe. Uses Joblib for parallelization.
        """
        groups = df.groupby([self.id_col])
        scaled_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE, backend=backend, timeout=timeout)(delayed(self.df_scaler)(gdf) for _, gdf in groups)
        gdf = pd.concat(scaled_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        get_reusable_executor().shutdown(wait=True)

        return gdf

    def df_scaler(self, gdf):
        """
        Scale covariates for lowest_key_level only
        """
        # obtain scalers
        
        scale_gdf = gdf.reset_index(drop=True)

        if scale_gdf['key_level'].unique().tolist()[0] == self.covar_key_level:

            # for lowest level keys, scale both target & co-variates
            if self.scaling_method == 'mean_scaling':

                if len(self.temporal_known_num_col_list) > 0:
                    known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                    known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
                    known_scale = np.divide(known_sum, known_nz_count) + 1.0
                else:
                    known_scale = 1.0

            elif self.scaling_method == 'no_scaling':
                known_scale = 1.0

            # reset index
            gdf = gdf.reset_index(drop=True)

            # scale each feature independently
            if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
                gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list]/known_scale

        return gdf

    def sort_dataset(self, data):
        """
        sort pandas dataframe by provided col list & order
        """
        if len(self.sort_col_list) > 0:
            data = data.sort_values(by=self.sort_col_list, ascending=True)
        else:
            pass
        return data
    
    def get_key_weights(self, data):
        """
        obtain weights for each id for weighted training option
        """
        if self.wt_col is None:
            data['Key_Sum'] = data[data[self.time_index_col] <= self.test_till].groupby(self.id_col)[self.target_col].transform(lambda x: x.sum())
            data['Key_Sum'] = data.groupby(self.id_col)['Key_Sum'].ffill()
            data['Key_Weight'] = data['Key_Sum']/data[data[self.time_index_col] <= self.test_till][self.target_col].sum()
        else:
            data['Key_Weight'] = data[self.wt_col]
            data['Key_Weight'] = data.groupby(self.id_col)['Key_Weight'].ffill()

        # 07/01/24
        #data['Key_Weight'] = data['Key_Weight']/data['Key_Weight'].max()
        #wt_median = data['Key_Weight'].median()
        #data['Key_Weight'] = data['Key_Weight'].clip(lower=wt_median)

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
        
        onehot_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + self.global_context_col_list
        df = pd.concat([df[onehot_col_list], pd.get_dummies(data=df, columns=onehot_col_list, prefix_sep='_')], axis=1, join='inner')
        
        return df

    def create_lead_lag_features(self, df):

        self.node_features_label = {}
        self.lead_lag_features_dict = {}

        for col in [self.target_col] + \
                   self.temporal_known_num_col_list + \
                   self.temporal_unknown_num_col_list + \
                   self.known_onehot_cols + \
                   self.unknown_onehot_cols:

            # instantiate with empty lists
            self.lead_lag_features_dict[col] = []

            for lag in range(self.max_lags, 0, -1):
                df[f'{col}_lag_{lag}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=lag, fill_value=0)
                self.lead_lag_features_dict[col].append(f'{col}_lag_{lag}')

            if col in self.temporal_known_num_col_list + self.known_onehot_cols:

                for lead in range(0, self.max_leads):
                    df[f'{col}_lead_{lead}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=-lead, fill_value=0)
                    self.lead_lag_features_dict[col].append(f'{col}_lead_{lead}')

            self.node_features_label[col] = self.lead_lag_features_dict[col]

        # don't drop rows with NaNs in lag/lead cols
        self.all_lead_lag_cols = list(itertools.chain.from_iterable([feat_col_list for col, feat_col_list in self.lead_lag_features_dict.items()]))

        return df

    def pad_dataframe(self, df, dateindex):
        # this ensures num nodes in a graph don't change from period to period. Essentially, we introduce dummy nodes.
        
        # function to fill NaNs in group id & stat cols post padding
        def fillgrpid(x):
            id_val = x[self.id_col].unique().tolist()[0]
            x = dateindex.merge(x, on=[self.time_index_col], how='left').fillna({self.id_col: id_val})
            
            for col in self.global_context_col_list + self.global_context_onehot_cols + ['key_level', 'key_list', 'Key_Weight']:
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

        # "padded" dataset with padding constant used as nan filler
        df = df.groupby(self.id_col, sort=False).apply(lambda x: fillgrpid(x).fillna(self.pad_constant)).reset_index(drop=True)

        # align datatypes within columns; some columns may have mixed types as a result of pad_constant
        for col, datatype in df.dtypes.to_dict().items():
            if col != 'Unnamed: 0':
                if datatype == 'object':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(datatype)
                    
        # convert all bool columns to 1/0 (problem on linux only)
        for col in self.known_onehot_cols+self.unknown_onehot_cols:
            if df[col].dtypes.name == 'bool':
                df[col] = df[col]*1.0
            else:
                df[col] = df[col].astype(str).astype(bool).astype(int)
        
        return df
                
    def parallel_pad_dataframe(self, df):
        """
        Individually pad each key
        """
        # get a df of all timestamps in the dataset
        dateindex = pd.DataFrame(sorted(df[self.time_index_col].unique()), columns=[self.time_index_col])

        groups = df.groupby([self.id_col])
        padded_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE, backend=backend, timeout=timeout)(delayed(self.pad_dataframe)(gdf, dateindex) for _, gdf in groups)
        gdf = pd.concat(padded_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        get_reusable_executor().shutdown(wait=True)
        return gdf

    def get_relative_time_index(self, df):
        """
        Obtain a numeric feature indicating recency of a timestamp. This feature is also used to impl. recency_weights.
        Ensure to run this after dataframe padding.
        """
        num_unique_ts = int(df[self.time_index_col].nunique())
        df['relative_time_index'] = df.groupby(self.id_col)[self.time_index_col].transform(lambda x: np.arange(num_unique_ts))
        df['relative_time_index'] = df['relative_time_index']/num_unique_ts
        df['recency_weights'] = np.exp(self.recency_alpha * df['relative_time_index'])

        return df

    def preprocess(self, data):
        
        print("   preprocessing dataframe - check for null columns...")
        # check null
        null_status, null_cols = self.check_null(data)
        
        if null_status:
            print("NaN column(s): ", null_cols)
            raise ValueError("Column(s) with NaN detected!")
            
        # get weights
        #print("   preprocessing dataframe - get id weights...")
        #df = self.get_key_weights(data)

        # create new keys
        print("   preprocessing dataframe - creating aggregate keys...")
        df = self.create_new_keys(data)

        # create new targets
        print("   preprocessing dataframe - creating new targets for aggregate keys...")
        df = self.create_new_targets(df)

        # create keybom
        print("   preprocessing dataframe - creating key bom...")
        df_keybom = self.get_keybom(df)

        # stack subkey level dfs into one df
        print("   preprocessing dataframe - consolidating all keys into one df...")
        df = self.stack_key_level_dataframes(df, df_keybom)

        # sort
        print("   preprocessing dataframe - sort by datetime & id...")
        df = self.sort_dataset(df)

        # scale dataset
        print("   preprocessing dataframe - scale target...")
        df = self.scale_target(df)
        print("   preprocessing dataframe - scale numeric known cols...")
        df = self.scale_covariates(df)

        # onehot encode
        print("   preprocessing dataframe - onehot encode categorical columns...")
        df = self.onehot_encode(df)

        print("   preprocessing dataframe - gather node specific feature cols...")
        # node types & node features
        self.node_cols = [self.target_col] + self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
        
        self.node_features = {}
        self.global_context_onehot_cols = []
        self.known_onehot_cols = []
        self.unknown_onehot_cols = []

        for node in self.global_context_col_list:
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
            
        self.temporal_nodes = self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

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
    
    def create_snapshot_graph(self, df_snap, period):
        # index nodes
        col_map_dict = self.node_indexing(df_snap, [self.id_col]+self.static_cat_col_list+self.global_context_col_list)
        
        # map id to indices
        for col, id_map in col_map_dict.items():
            df_snap[col] = df_snap[col].map(id_map["index"]).astype(int)

        # convert 'key_list' to key indices
        df_snap = df_snap.assign(mapped_key_list=[[col_map_dict[self.id_col]['index'][k] for k in literal_eval(row) if col_map_dict[self.id_col]['index'].get(k)] for row in df_snap['key_list']])
        df_snap['mapped_key_list_arr'] = df_snap['mapped_key_list'].apply(lambda x: np.array(x))
        keybom_nested = torch.nested.nested_tensor(list(df_snap['mapped_key_list_arr'].values), dtype=torch.int64, requires_grad=False)
        keybom_padded = torch.nested.to_padded_tensor(keybom_nested, -1)

        # Create HeteroData Object
        data = HeteroData({"y_mask": None, "y_weight": None})
        
        # get node features
        data[self.target_col].x = torch.tensor(df_snap[self.lead_lag_features_dict[self.target_col]].to_numpy(), dtype=torch.float)
        data[self.target_col].y = torch.tensor(df_snap[self.target_col].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_weight = torch.tensor(df_snap['Key_Weight'].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_mask = torch.tensor(df_snap['y_mask'].to_numpy().reshape(-1, 1), dtype=torch.float)
        if self.recency_weights:
            data[self.target_col].recency_weight = torch.tensor(df_snap['recency_weights'].to_numpy().reshape(-1, 1), dtype=torch.float)

        # get keybom for index_select in the model
        data['keybom'].x = keybom_padded

        # get status of key based on whether key_level == covar_key_level
        data['key_aggregation_status'].x = torch.tensor(np.where(df_snap['key_level'] == self.covar_key_level, 0, 1).reshape(-1, 1), dtype=torch.int64)

        # store snapshot period
        data[self.target_col].time_attr = period
        
        for col in self.temporal_known_num_col_list:
            data[col].x = torch.tensor(df_snap[self.lead_lag_features_dict[col]].to_numpy(), dtype=torch.float)
            
        for col in self.temporal_unknown_num_col_list:
            data[col].x = torch.tensor(df_snap[self.lead_lag_features_dict[col]].to_numpy(), dtype=torch.float)
        
        for col in self.known_onehot_cols:
            data[col].x = torch.tensor(df_snap[self.lead_lag_features_dict[col]].to_numpy(), dtype=torch.float)
        
        for col in self.unknown_onehot_cols:
            data[col].x = torch.tensor(df_snap[self.lead_lag_features_dict[col]].to_numpy(), dtype=torch.float)
            
        # global context node features (one-hot features)
        for col in self.global_context_col_list:
            onehot_cols_prefix = str(col) + '_'
            onehot_col_features = [f for f in df_snap.columns.tolist() if f.startswith(onehot_cols_prefix)]
            feats_df = df_snap[onehot_col_features].drop_duplicates()
            data[col].x = torch.tensor(feats_df[onehot_col_features].to_numpy(), dtype=torch.float)
                
        # bidirectional edges between global context node & target_col nodes
        for col in self.global_context_col_list:
            col_unique_values = sorted(df_snap[col].unique().tolist())
            
            fwd_edges_stack = []
            rev_edges_stack = []
            for value in col_unique_values:
                # get subset of all nodes with common col value
                edges = df_snap[(df_snap[col] == value) & (df_snap['key_level'] == self.covar_key_level)][[self.id_col, col]].to_numpy()
                rev_edges = df_snap[(df_snap[col] == value) & (df_snap['key_level'] == self.covar_key_level)][[col, self.id_col]].to_numpy()
                fwd_edges_stack.append(edges)
                rev_edges_stack.append(rev_edges)
                    
            # fwd edges
            edges = np.concatenate(fwd_edges_stack, axis=0)
            edge_name = (self.target_col, 'hascontext_{}'.format(col), col)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            # reverse edges
            rev_edges = np.concatenate(rev_edges_stack, axis=0)
            rev_edge_name = (col, '{}_contextof'.format(col), self.target_col)
            data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)
            
        # bidirectional edges exist between target_col nodes related by various static cols

        # get all key levels
        key_levels = df_snap['key_level'].unique().tolist()

        # for each key_level created intra key_level edges
        intra_key_level_edges = {}

        for key_level in key_levels:

            for col in self.static_cat_col_list:
                col_unique_values = sorted(df_snap[col].unique().tolist())

                fwd_edges_stack = []
                rev_edges_stack = []
                for value in col_unique_values:
                    # get subset of all nodes with common col value
                    nodes = df_snap[(df_snap[col] == value) & (df_snap['key_level'] == key_level)][self.id_col].to_numpy()
                    # Build all combinations of connected nodes
                    permutations = list(itertools.combinations(nodes, 2))
                    intra_key_level_edges[key_level] = permutations
                    edges_source = [e[0] for e in permutations]
                    edges_target = [e[1] for e in permutations]
                    edges = np.column_stack([edges_source, edges_target])
                    rev_edges = np.column_stack([edges_target, edges_source])
                    fwd_edges_stack.append(edges)
                    rev_edges_stack.append(rev_edges)

                # edge names
                edge_name = (self.target_col, 'relatedby_{}_at_{}'.format(col, key_level), self.target_col)
                rev_edge_name = (self.target_col, 'rev_relatedby_{}_at_{}'.format(col, key_level), self.target_col)
                # add edges to Data()
                edges = np.concatenate(fwd_edges_stack, axis=0)
                rev_edges = np.concatenate(rev_edges_stack, axis=0)
                data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
                data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)

        # get inter key_level edges based on static col similarities, using the following algo:
        # 1. get all the possible edges between all the keys not in current key_level
        # 2. subtract the set of intra key_level edges, leaving us with only aggregation edges

        for key_level in key_levels:

            for col in self.static_cat_col_list:
                col_unique_values = sorted(df_snap[col].unique().tolist())

                fwd_edges_stack = []
                rev_edges_stack = []
                for value in col_unique_values:
                    # get subset of all nodes with common col value
                    nodes = df_snap[(df_snap[col] == value) & (df_snap['key_level'] != key_level)][self.id_col].to_numpy()
                    # Build all combinations of connected nodes
                    permutations = list(itertools.combinations(nodes, 2))
                    # remove intra key_level edges
                    for k, v in intra_key_level_edges.items():
                        if k != key_level:
                            permutations = list(set(permutations) - set(v))

                    edges_source = [e[0] for e in permutations]
                    edges_target = [e[1] for e in permutations]
                    edges = np.column_stack([edges_source, edges_target])
                    rev_edges = np.column_stack([edges_target, edges_source])
                    fwd_edges_stack.append(edges)
                    rev_edges_stack.append(rev_edges)

                # edge names
                edge_name = (self.target_col, 'aggregatedby_{}_minus_{}'.format(col, key_level), self.target_col)
                rev_edge_name = (self.target_col, 'rev_aggregatedby_{}_minus_{}'.format(col, key_level), self.target_col)
                # add edges to Data()
                edges = np.concatenate(fwd_edges_stack, axis=0)
                rev_edges = np.concatenate(rev_edges_stack, axis=0)
                data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
                data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)


        # static nodes only required in this kind of connection
        """
        for col in self.static_cat_col_list:
            feats_df = df_snap[[col]]
            feats_df[f'dummy_static_{col}'] = 1  # assign a constant as dummy feature
            feats_df = feats_df.drop_duplicates()
            data[col].x = torch.tensor(feats_df[[f'dummy_static_{col}']].to_numpy(), dtype=torch.float)
        """
        # directed edges are from co-variates to target
        
        for col in self.temporal_known_num_col_list+self.temporal_unknown_num_col_list+self.known_onehot_cols+self.unknown_onehot_cols:

            nodes = df_snap[df_snap['key_level'] == self.covar_key_level][self.id_col].to_numpy()
            edges = np.column_stack([nodes, nodes])
                
            edge_name = (col, '{}_effect'.format(col), self.target_col)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            
            if not self.directed_graph:
                rev_edge_name = (self.target_col, 'covar_embed_update_{}'.format(col), col)
                data[rev_edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
                
        # validate dataset
        print("validate snapshot graph ...")    
        data.validate(raise_on_error=True)
        
        return data

    def onetime_dataprep(self, df):
        # preprocess
        print("preprocessing dataframe...")
        df = self.preprocess(df)

        # pad dataframe if required (will return df unchanged if not)
        print("padding dataframe...")
        df = self.parallel_pad_dataframe(df)  # self.pad_dataframe(df)
        print("creating relative time index & recency weights...")
        self.onetime_prep_df = self.get_relative_time_index(df)
        # add 'relative time index' to self.temporal_known_num_col_list
        self.temporal_known_num_col_list = self.temporal_known_num_col_list + ['relative_time_index']

    def create_train_test_dataset(self, df):

        # create lagged features
        print("create lead & lag features...")
        df = self.create_lead_lag_features(df)

        # split into train,test,infer
        print("splitting dataframe for training & testing...")
        train_df, test_df = self.split_train_test(df)
        
        df_dict = {'train': train_df, 'test': test_df}
        
        def parallel_snapshot_graphs(df, period):
            df_snap = df[df[self.time_index_col] == period].reset_index(drop=True)
            snapshot_graph = self.create_snapshot_graph(df_snap, period)
            return snapshot_graph
        
        # for each split create graph dataset iterator
        print("gather snapshot graphs...")
        datasets = {}
        for df_type, df in df_dict.items():

            # sample from self.subgraph_samples_col for smaller graphs
            if self.subgraph_sample_col is not None:
                all_subgraph_col_values = df[self.subgraph_sample_col].unique().tolist()
                # shuffle
                random.shuffle(all_subgraph_col_values)
                # sample
                snapshot_list = []
                # all snapshot timestamps
                snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)
                # restrict samples for very large datasets based on interleaving
                if df_type == 'train':
                    snap_periods_list = snap_periods_list[int(self.max_lags - 1):]
                if (self.interleave > 1) and (df_type == 'train'):
                    snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]

                if self.subgraph_sample_size > 0:
                    for i in range(0, len(all_subgraph_col_values), int(self.subgraph_sample_size)):
                        df_sample = df[df[self.subgraph_sample_col].isin(all_subgraph_col_values[i:i+self.subgraph_sample_size])]
                        # sample snapshot graphs
                        print("  gathering for subgraph_col_values: ", all_subgraph_col_values[i:i+self.subgraph_sample_size])
                        sample_snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df_sample, period) for period in snap_periods_list)
                        snapshot_list.append(sample_snapshot_list)
                else:
                    # sample snapshot graphs
                    sample_snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS,
                                                    batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)
                    snapshot_list.append(sample_snapshot_list)

                # Create a dataset iterator
                print("total {} samples picked: {}".format(df_type, len(list(itertools.chain.from_iterable(snapshot_list)))))
                dataset = DataLoader(list(itertools.chain.from_iterable(snapshot_list)), batch_size=self.batch, shuffle=self.shuffle)

                # append
                datasets[df_type] = dataset
        
        train_dataset, test_dataset = datasets.get('train'), datasets.get('test')
        get_reusable_executor().shutdown(wait=True)

        return train_dataset, test_dataset

    def create_infer_dataset(self, df, infer_till):

        self.infer_till = infer_till

        # drop lead/lag features if present
        try:
            df.drop(columns=self.all_lead_lag_cols, inplace=True)
        except:
            pass

        # create lagged features
        print("create lead & lag features...")
        df = self.create_lead_lag_features(df)

        # split into train,test,infer
        infer_df = self.split_infer(df)

        df_dict = {'infer': infer_df}
        
        # for each split create graph dataset iterator
        datasets = {}
        for df_type, df in df_dict.items():
            # snapshot start period: time.min() + max_history + fh, end_period:
            
            snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)[-1]
            #print("inference snapshot period: ",snap_periods_list)
            
            # create individual snapshot graphs
            snapshot_list = []
            for period in [snap_periods_list]:
                df_snap = df[df[self.time_index_col]==period].reset_index(drop=True)
                snapshot_graph = self.create_snapshot_graph(df_snap, period)
                snapshot_list.append(snapshot_graph)
                # get node index map
                df_node_map_index = df[df[self.time_index_col]==period].reset_index(drop=True)
                self.node_index_map = self.node_indexing(df_node_map_index, [self.id_col])

            # Create a dataset iterator
            dataset = DataLoader(snapshot_list, batch_size=1, shuffle=False) 
            
            # append
            datasets[df_type] = dataset
        
        infer_dataset = datasets.get('infer')

        return infer_df, infer_dataset

    def split_train_test(self, data):
        
        train_data = data[data[self.time_index_col] <= self.train_till].reset_index(drop=True)
        test_data = data[(data[self.time_index_col] > self.train_till)&(data[self.time_index_col] <= self.test_till)].reset_index(drop=True)
        
        return train_data, test_data
    
    def split_infer(self, data):

        infer_data = data[data[self.time_index_col] <= self.infer_till].reset_index(drop=True)
        
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
        statistics['num_target_nodes'] = int(batch[self.target_col].num_nodes)
        statistics['num_edges'] = batch.num_edges
        statistics['node_feature_dims'] = batch.num_node_features
        statistics['num_target_features'] = batch[self.target_col].num_node_features
        statistics['forecast_dim'] = batch[self.target_col].y.shape[1]
        
        return statistics
      
    def process_output(self, infer_df, model_output):
       
        if not self.categorical_onehot_encoding:
            self.temporal_known_num_col_list = list(set(self.temporal_known_num_col_list) - set(self.label_encoded_col_list))
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.label_encoded_col_list))

        infer_df = infer_df.groupby(self.id_col, sort=False).apply(lambda x: x[-1:]).reset_index(drop=True)
        print(infer_df[self.time_index_col].unique().tolist())

        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            scaler_cols = ['scaler']
        elif self.scaling_method == 'quantile_scaling':
            scaler_cols = ['scaler_iqr', 'scaler_median']
        else:
            scaler_cols = ['scaler_mu', 'scaler_std']
        
        infer_df = infer_df[[self.id_col, 'key_level', self.target_col, self.time_index_col] + self.static_cat_col_list + self.global_context_col_list + scaler_cols]
        
        model_output = model_output.reshape(-1,1)
        output = pd.DataFrame(data=model_output, columns=['forecast'])
        
        # merge forecasts with infer df
        output = pd.concat([infer_df, output], axis=1)    

        return output
        
    def update_dataframe(self, df, output):
        
        # merge output & base_df
        reduced_output_df = output[[self.id_col, 'key_level', self.time_index_col, 'forecast']]
        df_updated = df.merge(reduced_output_df, on=[self.id_col, 'key_level', self.time_index_col], how='left')
        
        # update target for current ts with forecasts
        df_updated[self.target_col] = np.where(df_updated['forecast'].isnull(), df_updated[self.target_col], df_updated['forecast'])
        
        # drop forecast column
        df_updated = df_updated.drop(columns=['forecast'])
        
        return df_updated
    
    def build_dataset(self, df):
        # onetime prep
        self.onetime_dataprep(df)
        # build graph datasets for train/test
        self.train_dataset, self.test_dataset = self.create_train_test_dataset(self.onetime_prep_df)

    def build_infer_dataset(self, infer_till):
        # build graph datasets for infer
        try:
            del self.infer_dataset
            gc.collect()
        except:
            pass

        _, self.infer_dataset = self.create_infer_dataset(df=self.onetime_prep_df, infer_till=infer_till)

    def build(self,
              model_type = "SAGE", 
              model_option = "BASIC", 
              model_dim = 128,
              num_layers = 1,
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
        self.forecast_quantiles = forecast_quantiles
        sample_batch = next(iter(self.train_dataset))
        
        # target device to train on ['cuda','cpu']
        self.device = torch.device(device)
        
        # build model
        self.model = STGNN(model_type = model_type,
                           model_option = model_option,
                           hidden_channels = model_dim, 
                           heads = attention_heads,
                           out_channels = self.fh, 
                           metadata = self.metadata, 
                           target_node = self.target_col,
                           context_nodes = self.global_context_col_list,
                           device = self.device,
                           n_quantiles = max(len(self.forecast_quantiles), 1),
                           num_layers = num_layers,
                           alpha = 0.5, 
                           dropout = dropout,
                           residual_conn_type = residual_conn_type,
                           loss_type = 'Quantile', 
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
              loss_type='Quantile',
              delta=1.0,
              use_lr_scheduler=True, 
              scheduler_params={'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001},
              sample_weights=False):
        
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
                if not self.grad_accum:
                    optimizer.zero_grad()

                batch = batch.to(self.device)
                batch_size = batch.num_graphs
                out = self.model(batch.x_dict, batch.edge_index_dict)
                
                # compute loss masking out N/A targets -- last snapshot
                if self.loss_type == 'Quantile':
                    try:
                        loss = loss_fn.loss(out, batch[self.target_col].y)
                    except:
                        loss = loss_fn.loss(torch.unsqueeze(out, dim=1), batch[self.target_col].y)
                    mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)
                elif self.loss_type == 'Huber':
                    try:
                        loss = loss_fn(out[:, -1, :], batch[self.target_col].y)
                    except:
                        loss = loss_fn(out, batch[self.target_col].y)
                    mask = batch[self.target_col].y_mask
                else:
                    try:
                        loss = loss_fn.loss(out[:, -1, :], batch[self.target_col].y)
                    except:
                        loss = loss_fn.loss(out, batch[self.target_col].y)
                    mask = batch[self.target_col].y_mask
                
                if sample_weights:
                    wt = batch[self.target_col].y_weight
                else:
                    wt = 1

                if self.recency_weights:
                    recency_wt = batch[self.target_col].recency_weight
                else:
                    recency_wt = 1

                loss = torch.mean(loss*mask*wt*recency_wt)

                # normalize loss to account for batch accumulation
                if self.grad_accum:
                    loss = loss / self.accum_iter
                    loss.backward()
                    # weights update
                    if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_dataset)):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()

                total_examples += batch_size
                total_loss += float(loss)
                
            return total_loss / total_examples
        
        def test_fn():
            self.model.train(False)
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
                            loss = loss_fn.loss(out, batch[self.target_col].y)
                        except:
                            loss = loss_fn.loss(torch.unsqueeze(out, dim=1), batch[self.target_col].y)
                        mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)
                    elif self.loss_type == 'Huber':
                        try:
                            loss = loss_fn(out[:, -1, :], batch[self.target_col].y)
                        except:
                            loss = loss_fn(out, batch[self.target_col].y)
                        mask = batch[self.target_col].y_mask
                    else:
                        try:
                            loss = loss_fn.loss(out[:, -1, :], batch[self.target_col].y)
                        except:
                            loss = loss_fn.loss(out, batch[self.target_col].y)
                        mask = batch[self.target_col].y_mask
                    
                    if sample_weights:
                        wt = batch[self.target_col].y_weight
                    else:
                        wt = 1

                    if self.recency_weights:
                        recency_wt = batch[self.target_col].recency_weight
                    else:
                        recency_wt = 1
                    
                    loss = torch.mean(loss*mask*wt*recency_wt)
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
                print("clearing gpu memory: ")
                del self.train_dataset, self.test_dataset
                gc.collect()
                torch.cuda.empty_cache()

                break

    def change_device(self, device='cpu'):
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(self.best_model, map_location=self.device))

    def disable_cuda_backend(self,):
        self.change_device(device="cuda")
        torch.backends.cudnn.enabled = False

    def infer(self, infer_start, infer_end, select_quantile, compute_mape=False):
        
        base_df = self.onetime_prep_df.copy()

        # get list of infer periods
        infer_periods = sorted(base_df[(base_df[self.time_index_col] >= infer_start) & (base_df[self.time_index_col] <= infer_end)][self.time_index_col].unique().tolist())
        
        # print model used for inference
        print("running inference using best saved model: ", self.best_model)
        
        forecast_df = pd.DataFrame() 
        
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

        for i,t in enumerate(infer_periods):
            
            print("forecasting period {} at lag {}".format(t, i))
            
            if not self.categorical_onehot_encoding:
                self.temporal_known_num_col_list = list(set(self.temporal_known_num_col_list) - set(self.label_encoded_col_list))
                self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.label_encoded_col_list))
        
            # infer dataset creation 
            infer_df, infer_dataset = self.create_infer_dataset(base_df, infer_till=t)
            output = infer_fn(self.model, self.best_model, infer_dataset)

            del infer_dataset
            gc.collect()
            torch.cuda.empty_cache()

            # select output quantile
            output_arr = output[0]
            output_arr = output_arr.cpu().numpy()
            
            # quantile selection
            min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)
            
            if self.loss_type == 'Quantile':
                assert select_quantile >= min_qtile and select_quantile <= max_qtile, "selected quantile out of bounds!"
                try:
                    q_index = self.forecast_quantiles(select_quantile)
                    output_arr = output_arr[:,:,q_index] 
                except:
                    q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                    q_lower = int(q_upper - 1)
                    q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower] )/(self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                    q_lower_weight = 1 - q_upper_weight
                    output_arr = q_upper_weight*output_arr[:,:,q_upper] + q_lower_weight*output_arr[:,:,q_lower]
            else:
                try:
                    output_arr = output_arr[:, :, 0]
                except:
                    pass
                
            # show current o/p
            output = self.process_output(infer_df, output_arr)

            # append forecast
            forecast_df = forecast_df.append(output)

            # update df
            base_df = self.update_dataframe(base_df, output)

        # re-scale output

        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler']
        elif self.scaling_method == 'quantile_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
        else:
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_std'] + forecast_df['scaler_mu']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_std'] + forecast_df['scaler_mu']

        return forecast_df
