#!/usr/bin/env python
# coding: utf-8
import random

# Model Specific imports
import torch
import copy
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, HeteroConv, SAGEConv, BatchNorm, LayerNorm, HANConv, HGTConv, GATv2Conv, aggr
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import simplified.CustomLayers as CustomLayers
import gc
import math

# Data specific imports
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

# Tweedie Specific
import statsmodels.api as sm
import scipy as sp
from tweedie import tweedie

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
import os, psutil
import time

# timeout specific imports
import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

########################


os_type = sys.platform


if os_type == 'linux':
    backend = 'loky'
    timeout = 3600
else:
    backend = 'threading'
    timeout = 3600

# set default dtype to float32
torch.set_default_dtype(torch.float32)


# Models & Utils

# loss function


class QuantileLoss:
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


class RMSE:
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """
    def __init__(self):
        super().__init__()

    def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:
        loss = torch.pow(y_pred - target, 2)
        return loss


class TweedieLoss:
    def __init__(self, p_list=[]):
        super().__init__()
        self.p_list = p_list

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, p, scaler, log1p_transform):
        # convert all 2-d inputs to 3-d
        y_true = torch.unsqueeze(y_true, dim=2)
        scaler = torch.unsqueeze(scaler, dim=2)

        if len(self.p_list) > 0:
            pass
        else:
            self.p_list.append(p)

        """
        if log1p_transform:
            # log1p first, scale next
            # reverse process actual
            y_true = torch.expm1(y_true * scaler)
            # reverse proceess y_pred
            y_pred = torch.expm1(torch.exp(y_pred) * scaler)
            # take log of y_pred again
            y_pred = torch.log(y_pred + 1e-8)

            a = y_true * torch.exp(y_pred * (1 - p)) / (1 - p)
            b = torch.exp(y_pred * (2 - p)) / (2 - p)
            loss = -a + b

        """
        if log1p_transform:
            # scale first, log1p after
            y_true = torch.expm1(y_true) * scaler
            # reverse log of prediction y_pred
            y_pred = torch.exp(y_pred)
            # clamp predictions
            y_pred = torch.clamp(y_pred, min=-7, max=7)
            # get pred
            y_pred = torch.expm1(y_pred) * scaler
            # take log of y_pred again
            y_pred = torch.log(y_pred + 1e-8)

            loss = 0
            for pn in self.p_list:
                pn = torch.unsqueeze(pn, dim=2)
                a = y_true * torch.exp(y_pred * (1 - pn)) / (1 - pn)
                b = torch.exp(y_pred * (2 - pn)) / (2 - pn)
                loss += (-a + b)
        else:
            # no log1p
            # clamp predictions
            y_pred = torch.clamp(y_pred, min=-7, max=7)
            y_true = y_true * scaler
            loss = 0
            for pn in self.p_list:
                pn = torch.unsqueeze(pn, dim=2)
                a = y_true * torch.exp((y_pred + torch.log(scaler)) * (1 - pn)) / (1 - pn)
                b = torch.exp((y_pred + torch.log(scaler)) * (2 - pn)) / (2 - pn)
                loss += (-a + b)

            """
            a = y_true * torch.exp(y_pred * (1 - p)) / (1 - p)
            b = torch.exp(y_pred * (2 - p)) / (2 - p)
            loss = -a + b
            """
        return loss


class Poisson:
    """
    Poisson NLL Loss

    """
    def __init__(self):
        super().__init__()

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = torch.unsqueeze(y_true, dim=2)
        y_pred = torch.exp(y_pred)
        loss = torch.nn.functional.poisson_nll_loss(input=y_true,
                                                    target=y_pred,
                                                    log_input=False,
                                                    full=False,
                                                    eps=1e-08,
                                                    reduction='none')
        return loss


class SMAPE:
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Compute SMAPE loss between predictions and ground truth values.

        Parameters:
            y_true (torch.Tensor): Ground truth values.
            y_pred (torch.Tensor): Predicted values.
            epsilon: prevent 0 division when y_pred, y_true are 0
        Returns:
            torch.Tensor: SMAPE loss.
        """
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.clamp(torch.abs(y_pred) + torch.abs(y_true), min=self.epsilon)
        loss = 2.0 * (numerator / denominator)

        return loss


class Huber:
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        loss = torch.nn.functional.huber_loss(input=y_pred, target=y_true, reduction='none', delta=self.delta)
        return loss


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
            self.lin_self(x) + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) +
            self.alpha * self.conv_dst_to_src(x, edge_index)
                )


# Forecast GNN Layers
class HeteroGATv2Conv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads,
                 dropout,
                 edge_types,
                 node_types,
                 target_node_type,
                 first_layer,
                 is_output_layer=False):

        super().__init__()

        self.target_node_type = target_node_type
        self.node_types = node_types

        conv_dict = {}
        for e in edge_types:
            if e[0] == e[2]:
                conv_dict[e] = GATv2Conv(in_channels=-1,
                                         out_channels=out_channels,
                                         heads=heads,
                                         concat=False,
                                         add_self_loops=True,
                                         dropout=dropout,
                                         aggr='mean'
                                         )
            else:
                if first_layer:
                    if e[0] == e[2]:
                        conv_dict[e] = GATv2Conv(in_channels=-1,
                                                 out_channels=out_channels,
                                                 heads=heads,
                                                 concat=False,
                                                 add_self_loops=True,
                                                 dropout=dropout,
                                                 aggr='mean'
                                                 )
                    else:
                        conv_dict[e] = GATv2Conv(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 heads=heads,
                                                 concat=False,
                                                 add_self_loops=False,
                                                 dropout=dropout,
                                                 aggr='sum'
                                                 )

        self.conv = HeteroConv(conv_dict)

        if not is_output_layer:
            self.dropout = torch.nn.Dropout(dropout)
            self.norm_dict = torch.nn.ModuleDict({
                node_type:
                    LayerNorm(out_channels, mode='node')
                for node_type in node_types if node_type == target_node_type
            })

        self.is_output_layer = is_output_layer

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)

        if not self.is_output_layer:
            for node_type, norm in self.norm_dict.items():
                x_dict[node_type] = norm(self.dropout(x_dict[node_type]).relu())
        else:
            x_dict[self.target_node_type] = x_dict[self.target_node_type].relu()

        return x_dict


class HeteroForecastSageConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 edge_types,
                 node_types,
                 target_node_type,
                 first_layer,
                 is_output_layer=False):

        super().__init__()

        self.target_node_type = target_node_type

        conv_dict = {}
        for e in edge_types:
            if e[0] == e[2]:
                conv_dict[e] = SAGEConv(in_channels=-1,
                                        out_channels=out_channels,
                                        aggr='mean',
                                        project=False,
                                        normalize=False,
                                        bias=True)
            else:
                if first_layer:
                    if e[0] == e[2]:
                        conv_dict[e] = SAGEConv(in_channels=-1,
                                                out_channels=out_channels,
                                                aggr='mean',
                                                project=False,
                                                normalize=False,
                                                bias=True)
                    else:
                        conv_dict[e] = SAGEConv(in_channels=in_channels,
                                                out_channels=out_channels,
                                                aggr='sum',
                                                project=False,
                                                normalize=False,
                                                bias=True)

        self.conv = HeteroConv(conv_dict)

        if not is_output_layer:
            self.dropout = torch.nn.Dropout(dropout)
            self.norm_dict = torch.nn.ModuleDict({
                node_type:
                    LayerNorm(out_channels, mode='node')
                for node_type in node_types if node_type == target_node_type
            })

        self.is_output_layer = is_output_layer

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv(x_dict, edge_index_dict)

        if not self.is_output_layer:
            for node_type, norm in self.norm_dict.items():
                x_dict[node_type] = norm(self.dropout(x_dict[node_type]).relu())
        else:
            x_dict[self.target_node_type] = x_dict[self.target_node_type].relu()

        return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_rnn_layers, out_channels, dropout, node_types, edge_types,
                 target_node_type, skip_connection=True):
        super().__init__()

        self.target_node_type = target_node_type
        self.skip_connection = skip_connection
        self.num_layers = num_layers
        self.num_rnn_layers = num_rnn_layers

        if num_layers == 1:
            self.skip_connection = False

        """
        # linear projection
        self.node_proj = torch.nn.ModuleDict()
        for node_type in node_types:
            self.node_proj[node_type] = Linear(-1, hidden_channels)
        """

        self.transformed_feat_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            if node_type == target_node_type:
                self.transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                      hidden_size=hidden_channels,
                                                                      num_layers=self.num_rnn_layers,
                                                                      batch_first=True)
            else:
                self.transformed_feat_dict[node_type] = Linear(-1, hidden_channels)

        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroForecastSageConv(in_channels=in_channels if i == 0 else hidden_channels,
                                          out_channels=hidden_channels,
                                          dropout=dropout,
                                          node_types=node_types,
                                          edge_types=edge_types,
                                          target_node_type=target_node_type,
                                          first_layer=i == 0,
                                          is_output_layer=i == num_layers - 1,
                                          )

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        # transform target node
        for node_type, x in x_dict.items():
            if node_type == self.target_node_type:
                o, _ = self.transformed_feat_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3 -d (N,L,1)
                x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
            else:
                x_dict[node_type] = self.transformed_feat_dict[node_type](x)

        """
        # Linear project nodes
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_proj[node_type](x).relu()
        """
        """
        if self.skip_connection:
            res_dict = x_dict
        """

        # run convolutions
        for i, conv in enumerate(self.conv_layers):
            x_dict = conv(x_dict, edge_index_dict)

            """
            # apply skip connections every 4 layers
            if ((i + 1) % 2 == 0) and self.skip_connection:
                res_dict = {key: res_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x) in zip(x_dict.items(), res_dict.items()) if key == res_key}

            # update res input every 2 layers
            if ((i + 1) % 2 == 0) and self.skip_connection:
                res_dict = x_dict

            x_dict = {key: x.relu() for key, x in x_dict.items()}
            """

        return x_dict[self.target_node_type]


class HeteroGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, heads, node_types, edge_types,
                 target_node_type, skip_connection=True):
        super().__init__()

        self.target_node_type = target_node_type
        self.skip_connection = skip_connection
        self.num_layers = num_layers

        """
        # linear projection
        self.node_proj = torch.nn.ModuleDict()
        for node_type in node_types:
            self.node_proj[node_type] = Linear(-1, hidden_channels)
        """

        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroGATv2Conv(in_channels=in_channels if i == 0 else hidden_channels,
                                   out_channels=hidden_channels,
                                   dropout=dropout,
                                   heads=heads,
                                   node_types=node_types,
                                   edge_types=edge_types,
                                   target_node_type=target_node_type,
                                   first_layer=i == 0,
                                   is_output_layer=False)

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):
        """
        # Linear project nodes
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_proj[node_type](x).relu()
        """

        # run convolutions
        for i, conv in enumerate(self.conv_layers):
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict[self.target_node_type]


# HAN Models
class BasicHAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_rnn_layers, metadata, target_node_type, hidden_channels=128, heads=1,
                 dropout=0.1):
        super().__init__()
        self.target_node_type = target_node_type
        node_types = metadata[0]

        self.transformed_feat_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            if node_type == target_node_type:
                self.transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                      hidden_size=hidden_channels,
                                                                      num_layers=num_rnn_layers,
                                                                      batch_first=True)
            else:
                self.transformed_feat_dict[node_type] = Linear(-1, hidden_channels)

        # Conv Layer
        self.conv = HANConv(in_channels=in_channels,
                            out_channels=hidden_channels,
                            heads=heads,
                            dropout=dropout,
                            metadata=metadata)

    def forward(self, x_dict, edge_index_dict):

        # transform target node
        for node_type, x in x_dict.items():
            if node_type == self.target_node_type:
                o, _ = self.transformed_feat_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3 -d (N,L,1)
                x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
            else:
                x_dict[node_type] = self.transformed_feat_dict[node_type](x)

        x_dict = self.conv(x_dict, edge_index_dict)

        return x_dict[self.target_node_type]


class HAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, num_rnn_layers, target_node_type, hidden_channels=128, heads=1,
                 dropout=0.1):
        super().__init__()
        self.target_node_type = target_node_type

        # Conv Layers
        self.conv = CustomLayers.ModHANConv(in_channels=in_channels,
                                            out_channels=hidden_channels,
                                            heads=heads,
                                            dropout=dropout,
                                            metadata=metadata,
                                            target_node_type=target_node_type,
                                            num_rnn_layers=num_rnn_layers)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)

        return x_dict[self.target_node_type]


class SageHAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_rnn_layers, target_node_type, node_types, edge_types,
                 hidden_channels=128, heads=1, dropout=0.1):
        super().__init__()
        self.target_node_type = target_node_type
        target_edge_types = [edge_type for edge_type in edge_types if
                             (edge_type[0] == target_node_type) and (edge_type[2] == target_node_type)]

        self.transformed_feat_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            if node_type == target_node_type:
                self.transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                      hidden_size=hidden_channels,
                                                                      num_layers=num_rnn_layers,
                                                                      batch_first=True)
            else:
                self.transformed_feat_dict[node_type] = Linear(-1, hidden_channels)

        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        # get node embeddings from sage
        sage_conv = HeteroForecastSageConv(in_channels=(-1, -1),
                                           out_channels=hidden_channels,
                                           dropout=dropout,
                                           node_types=node_types,
                                           edge_types=edge_types,
                                           target_node_type=target_node_type,
                                           first_layer=True,
                                           is_output_layer=False)

        self.conv_layers.append(sage_conv)

        han_metadata = ([self.target_node_type], target_edge_types)

        han_conv = HANConv(in_channels=-1,
                           out_channels=hidden_channels,
                           heads=heads,
                           dropout=dropout,
                           metadata=han_metadata)

        self.conv_layers.append(han_conv)

    def forward(self, x_dict, edge_index_dict):

        # transform target node
        for node_type, x in x_dict.items():
            if node_type == self.target_node_type:
                o, _ = self.transformed_feat_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3 -d (N,L,1)
                x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
            else:
                x_dict[node_type] = self.transformed_feat_dict[node_type](x)

        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items() if key == self.target_node_type}
            edge_index_dict = {key: x for key, x in edge_index_dict.items() if (key[0] == self.target_node_type) and
                               (key[2] == self.target_node_type)}

        return x_dict[self.target_node_type]


class HGT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, target_node_type, heads=1, num_layers=1):
        super().__init__()
        self.target_node_type = target_node_type
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        target_edge_types = [edge_type for edge_type in metadata[1] if
                             (edge_type[0] == target_node_type) and (edge_type[2] == target_node_type)]
        partial_metadata = [[target_node_type], target_edge_types]

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata,
                               heads=heads)
            else:
                conv = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=partial_metadata,
                               heads=heads)

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        x_dict = {node_type: self.lin_dict[node_type](x).relu() for node_type, x in x_dict.items()}

        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items() if key == self.target_node_type}
            edge_index_dict = {key: x for key, x in edge_index_dict.items() if
                               (key[0] == self.target_node_type) and (key[2] == self.target_node_type)}

        return x_dict[self.target_node_type]


# Models

class STGNN(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_layers,
                 num_rnn_layers,
                 metadata,
                 target_node,
                 time_steps=1,
                 n_quantiles=1,
                 heads=1,
                 dropout=0.0,
                 tweedie_out=False,
                 skip_connection=True,
                 layer_type='HAN'):

        super(STGNN, self).__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.time_steps = time_steps
        self.n_quantiles = n_quantiles
        self.tweedie_out = tweedie_out
        self.layer_type = layer_type
        self.num_rnn_layers = num_rnn_layers

        # lstm & projection layers
        self.sequence_layer = torch.nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels,
                                            num_layers=num_rnn_layers, batch_first=True)
        self.projection_layer = Linear(hidden_channels, self.n_quantiles)

        if layer_type == 'SAGE':
            self.gnn_model = HeteroGraphSAGE(in_channels=(-1, -1),
                                             hidden_channels=hidden_channels,
                                             num_layers=num_layers,
                                             num_rnn_layers=num_rnn_layers,
                                             out_channels=hidden_channels,
                                             dropout=dropout,
                                             node_types=self.node_types,
                                             edge_types=self.edge_types,
                                             target_node_type=target_node,
                                             skip_connection=skip_connection)
        elif layer_type == 'HAN':
            self.gnn_model = HAN(in_channels=-1,
                                 out_channels=hidden_channels,
                                 num_rnn_layers=num_rnn_layers,
                                 metadata=metadata,
                                 target_node_type=target_node,
                                 hidden_channels=hidden_channels,
                                 heads=heads,
                                 dropout=dropout)
        elif layer_type == 'BasicHAN':
            self.gnn_model = BasicHAN(in_channels=-1,
                                      out_channels=hidden_channels,
                                      num_rnn_layers=num_rnn_layers,
                                      metadata=metadata,
                                      target_node_type=target_node,
                                      hidden_channels=hidden_channels,
                                      heads=heads,
                                      dropout=dropout)
        elif layer_type == 'SageHAN':
            self.gnn_model = SageHAN(in_channels=-1,
                                     out_channels=hidden_channels,
                                     num_rnn_layers=num_rnn_layers,
                                     target_node_type=target_node,
                                     node_types=self.node_types,
                                     edge_types=self.edge_types,
                                     hidden_channels=hidden_channels,
                                     heads=heads,
                                     dropout=dropout)

        elif layer_type == 'HGT':
            self.gnn_model = HGT(in_channels=-1,
                                 out_channels=hidden_channels,
                                 metadata=metadata,
                                 target_node_type=target_node,
                                 num_layers=num_layers,
                                 hidden_channels=hidden_channels,
                                 heads=heads)

        elif layer_type == 'GAT':
            self.gnn_model = HeteroGAT(in_channels=(-1, -1),
                                       hidden_channels=hidden_channels,
                                       num_layers=num_layers,
                                       out_channels=hidden_channels,
                                       dropout=dropout,
                                       heads=heads,
                                       node_types=self.node_types,
                                       edge_types=self.edge_types,
                                       target_node_type=target_node,
                                       skip_connection=skip_connection)

    def sum_over_index(self, x,  x_wt, x_index):
        # re-scale outputs
        x = torch.mul(x, x_wt)
        return torch.index_select(x, 0, x_index).sum(dim=0)

    def log_transformed_sum_over_index(self, x, x_wt, x_index):
        """
        For tweedie, the output is expected to be the log of required prediction, so, reverse log transform before aggregating.
        """
        x = torch.exp(x)
        x = torch.mul(x, x_wt)
        return torch.index_select(x, 0, x_index).sum(dim=0)

    def forward(self, x_dict, edge_index_dict):
        # get keybom
        keybom = x_dict['keybom']
        keybom = keybom.type(torch.int64)
        scaler = x_dict['scaler']

        # get key_aggregation_status
        key_agg_status = x_dict['key_aggregation_status']
        agg_indices = (key_agg_status == 1).nonzero(as_tuple=True)[0].tolist()

        # del keybom from x_dict
        del x_dict['keybom']
        del x_dict['key_aggregation_status']
        del x_dict['scaler']

        # gnn model
        gnn_embedding = self.gnn_model(x_dict, edge_index_dict)

        # get device
        device_int = gnn_embedding.get_device()
        if device_int == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        # loop over rest of time steps
        inp = gnn_embedding
        out_list = []
        h = torch.zeros((self.num_rnn_layers, gnn_embedding.shape[0], gnn_embedding.shape[1])).to(device)
        c = torch.zeros((self.num_rnn_layers, gnn_embedding.shape[0], gnn_embedding.shape[1])).to(device)
        for _ in range(self.time_steps):
            o, (h, c) = self.sequence_layer(torch.unsqueeze(inp, dim=1), (h, c))
            out_list.append(o[:, -1, :])
            inp = torch.add(inp, o[:, -1, :])

        out = torch.stack(out_list, dim=1)  # (n_nodes, ts, hidden_channels)
        # projection
        out = self.projection_layer(out)  # (n_nodes, ts, n_quantiles)

        # fallback to this approach (slower) in case vmap doesn't work
        # constrain the higher level key o/ps to be the sum of their constituents
        """
        for i in agg_indices:
            out[i] = torch.index_select(out, 0, keybom[i][keybom[i] != -1]).sum(dim=0)
        """

        # vectorized approach follows:

        dummy_out = torch.zeros(1, out.shape[1], out.shape[2]).to(device)
        # add a zero vector to the out tensor as workaround to the limitation of vmap of not being able to process
        # nested/dynamic shape tensors
        out = torch.cat([out, dummy_out], dim=0)

        # add dummy scale for last row in out
        scaler = torch.unsqueeze(scaler, 2)
        dummy_scaler = torch.ones(1, scaler.shape[1], scaler.shape[2]).to(device)
        scaler = torch.cat([scaler, dummy_scaler], dim=0)

        # replace -1 from key bom with last dim in out
        #keybom[keybom == -1] = int(out.shape[0] - 1)

        if keybom.shape[-1] == 1:
            # for rare non-hierarchical cases
            keybom[keybom == -1] = int(0)
        elif keybom.shape[-1] == 0:
            keybom = torch.zeros((1, 1), dtype=torch.int64).to(device)
        else:
            keybom[keybom == -1] = int(out.shape[0] - 1)

        # call vmap on sum_over_index function
        if self.tweedie_out:
            batched_sum_over_index = torch.vmap(self.log_transformed_sum_over_index, in_dims=(None, None, 0), randomness='error')
            out = batched_sum_over_index(out, scaler, keybom)
            # scale back
            out = out / scaler[:-1]
            # again do the log_transform on the aggregates
            out = torch.log(out)
        else:
            batched_sum_over_index = torch.vmap(self.sum_over_index, in_dims=(None, None, 0), randomness='error')
            out = batched_sum_over_index(out, scaler, keybom)
            # scale back
            out = out / scaler[:-1]

        # returned shape of out should be same as that before cat with dummy_out

        return out


# Graph Object

class graphmodel:
    def __init__(self, 
                 col_dict,
                 max_target_lags,
                 max_covar_lags,
                 max_leads,
                 train_till,
                 test_till,
                 test_recent_percentage=None,
                 test_random_percentage=None,
                 autoregressive_target=True,
                 lag_offset=0,
                 rolling_features_list=[],
                 min_history=1,
                 fh=1,
                 batch=1,
                 grad_accum=False,
                 accum_iter=1,
                 scaling_method='mean_scaling',
                 log1p_transform=False,
                 tweedie_out=False,
                 estimate_tweedie_p=False,
                 tweedie_p_range=[1.01, 1.95],
                 tweedie_variance_power=1.1,
                 iqr_high=0.75,
                 iqr_low=0.25,
                 categorical_onehot_encoding=True,
                 directed_graph=True,
                 shuffle=True,
                 interleave=1,
                 recency_weights=False,
                 recency_alpha=0,
                 output_clipping=False,
                 PARALLEL_DATA_JOBS=4,
                 PARALLEL_DATA_JOBS_BATCHSIZE=128):
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
        rolling_features_list: [('col','stat','periods','min_periods'), ...], 'stat' : ['mean','std']

        """
        super().__init__()
        
        self.col_dict = copy.deepcopy(col_dict)
        self.min_history = int(min_history)
        self.fh = int(fh)
        self.max_history = int(1)
        self.max_target_lags = int(max_target_lags) if (max_target_lags is not None) and (max_target_lags > 0) else 1
        self.max_covar_lags = int(max_covar_lags) if (max_covar_lags is not None) and (max_covar_lags > 0) else 1
        self.max_leads = int(max_leads) if (max_leads is not None) and (max_leads > 0) else 1

        # add offset to target_lags
        self.lag_offset = lag_offset
        self.max_target_lags = int(self.max_target_lags + lag_offset)

        assert self.max_leads >= self.fh, "max_leads must be >= fh"
        
        # adjust train_till/test_till for delta|max_leads - fh| in split_* methods
        self.train_till = train_till
        self.test_till = test_till
        self.test_recent_percentage = test_recent_percentage
        self.test_random_percentage = test_random_percentage
        self.autoregressive_target = autoregressive_target
        self.rolling_features_list = rolling_features_list
        
        self.batch = batch
        self.grad_accum = grad_accum
        self.accum_iter = accum_iter
        self.scaling_method = scaling_method
        self.log1p_transform = log1p_transform
        self.tweedie_out = tweedie_out
        self.estimate_tweedie_p = estimate_tweedie_p
        self.tweedie_p_range = tweedie_p_range
        self.tweedie_variance_power = tweedie_variance_power if isinstance(tweedie_variance_power, list) else [tweedie_variance_power]
        self.iqr_high = iqr_high
        self.iqr_low = iqr_low
        self.categorical_onehot_encoding = categorical_onehot_encoding
        self.directed_graph = directed_graph
        self.shuffle = shuffle
        self.interleave = interleave
        self.recency_weights = recency_weights
        self.recency_alpha = recency_alpha
        self.output_clipping = output_clipping
        self.PARALLEL_DATA_JOBS = PARALLEL_DATA_JOBS
        self.PARALLEL_DATA_JOBS_BATCHSIZE = PARALLEL_DATA_JOBS_BATCHSIZE
        self.pad_constant = 0

        # extract column sets from col_dict
        # hierarchy specific keys
        self.id_col = self.col_dict.get('id_col')
        self.key_combinations = self.col_dict.get('key_combinations')
        self.key_hierarchy = self.col_dict.get('key_hierarchy', None)
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

        # full column set for train/test/infer
        self.col_list = [self.id_col] + [self.target_col] + [self.time_index_col] + \
                         self.static_cat_col_list + self.global_context_col_list + \
                         self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                         self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.cat_col_list = self.global_context_col_list + self.static_cat_col_list + \
                            self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.multistep_target = []
        self.multistep_mask = []
        self.node_features_label = {}
        self.lead_lag_features_dict = {}
        self.all_lead_lag_cols = []
        self.node_cols = [self.target_col] + self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                         self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + \
                         self.global_context_col_list
        self.temporal_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + \
                                 self.temporal_known_num_col_list + self.temporal_unknown_num_col_list

        self.global_context_onehot_cols = []
        self.known_onehot_cols = []
        self.unknown_onehot_cols = []
        # scaler cols
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            self.scaler_cols = ['scaler']
        elif self.scaling_method == 'quantile_scaling':
            self.scaler_cols = ['scaler_iqr', 'scaler_median']
        else:
            self.scaler_cols = ['scaler_mu', 'scaler_std']

        # auto tweedie est
        if self.estimate_tweedie_p:
            self.tweedie_p_col = ['tweedie_p']
        else:
            self.tweedie_p_col = []

    def get_memory_usage(self, ):
        return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30, 2)

    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                        start_mem - end_mem) / start_mem))
        return df

    def check_data_sufficiency(self, df):
        """
        Exclude keys which do not have at least one data point within the training cutoff period
        """
        df = df.groupby(self.id_col).filter(lambda x: len(x[x[self.time_index_col] <= self.test_till]) >= self.min_history)

        return df

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

    def get_keybom_hierarchy(self, df):
        """
        For given hierarchy of key combinations, obtain list of constituent keys as keys in the level below
        """
        keybom_list = []
        for l1_key, l2_key in self.key_hierarchy.items():
            base_key = "key_" + "_".join(l1_key)
            agg_key = "key_" + "_".join(l2_key)
            df_key_map = df.groupby([agg_key, self.time_index_col])[base_key].apply(
                lambda x: x.unique().tolist()).rename('key_list').reset_index().rename(columns={agg_key: self.id_col})
            keybom_list.append(df_key_map)
        df_keybom = pd.concat(keybom_list, axis=0)
        df_keybom = df_keybom.reset_index(drop=True)

        return df_keybom

    def stack_key_level_dataframes(self, df, df_keybom):
        df_stack_list = []
        for (k, v), (k2, v2) in zip(self.key_levels_dict.items(), self.key_targets_dict.items()):
            if k == k2:
                if k == self.covar_key_level:
                    if self.wt_col is None:
                        df_temp = df[[k, v2,
                                      self.time_index_col] + v + self.global_context_col_list + self.static_cat_col_list + self.temporal_col_list]
                    else:
                        df_temp = df[[k, v2, self.time_index_col,
                                      self.wt_col] + v + self.global_context_col_list + self.static_cat_col_list + self.temporal_col_list]
                    df_temp = df_temp.drop_duplicates()
                else:
                    df_temp = df[
                        [k, v2, self.time_index_col] + v + self.global_context_col_list + self.static_cat_col_list]
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
        # Add 'Key_Level_Weight' col

        # normalized key level weight by key count
        key_levels_weight_dict = {}
        total_keys = df[self.id_col].nunique()
        for k in df['key_level'].unique().tolist():
            key_levels_weight_dict[k] = total_keys / df[df['key_level'] == k][self.id_col].nunique()

        df['Key_Level_Weight'] = df['key_level'].map(key_levels_weight_dict)
        print("Derived key_level weights: \n", key_levels_weight_dict)

        """
        # user assigned
        df['Key_Level_Weight'] = df['key_level'].map(self.key_levels_weight_dict)
        """

        # Add 'Key_Weight' col
        if self.wt_col is None:
            df['Key_Weight'] = 1
        else:
            df['Key_Weight'] = np.where(df['key_level'] == self.covar_key_level, df[self.wt_col], 1)
            # df['Key_Weight'] = np.where(df[self.wt_col].isnull(), 1, df[self.wt_col])
        # new col list
        self.col_list = df.columns.tolist()
        return df

    def power_optimization_loop(self, df):
        # initialization constants
        init_power = 1.01
        max_iterations = 100

        try:
            endog = df[self.target_col].astype(np.float32).to_numpy()
            exog = df[self.temporal_known_num_col_list].astype(np.float32).to_numpy()

            # fit glm model
            @exit_after(60)
            def glm_fit(endog, exog, power):
                res = sm.GLM(endog, exog,
                             family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=power)).fit()
                return res.mu, res.scale, res._endog

            # optimize 1 iter
            @exit_after(60)
            def optimize_power(res_mu, res_scale, power, res_endog):
                """
                returns optimized power as opt.x
                """

                def loglike_p(power):
                    return -tweedie(mu=res_mu, p=power, phi=res_scale).logpdf(res_endog).sum()

                try:
                    opt = sp.optimize.minimize_scalar(loglike_p, bounds=(1.02, 1.95), method='Bounded')
                except RuntimeWarning as e:
                    print(f'There was a RuntimeWarning: {e}')

                return opt.x

            # optimization loop
            power = init_power
            print("initializing with power: ", power)
            for i in range(max_iterations):

                res_mu, res_scale, res_endog = glm_fit(endog, exog, power)
                new_power = optimize_power(res_mu, res_scale, power, res_endog)

                # check if new power has converged
                if abs(new_power - power) >= 0.001:
                    power = new_power
                    print("iteration {}, updated power to: {}".format(i, power))
                else:
                    print("iteration {}, new_power unaccepted: {}".format(i, new_power))
                    break
            df['tweedie_p'] = round(power, 2)

        except:
            print("using default power of {} for {}".format(1.5, df[self.id_col].unique()))
            df['tweedie_p'] = 1.5

        # clip tweedie to within range
        df['tweedie_p'] = df['tweedie_p'].clip(lower=self.tweedie_p_range[0], upper=self.tweedie_p_range[1])

        return df

    def parallel_tweedie_p_estimate(self, df):
        """
        Individually obtain 'p' parameter for tweedie loss
        """
        groups = df.groupby([self.id_col])
        p_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE, backend=backend,
                          timeout=timeout)(delayed(self.power_optimization_loop)(gdf) for _, gdf in groups)
        gdf = pd.concat(p_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        get_reusable_executor().shutdown(wait=True)
        return gdf

    """ 
    def scale_target(self, df):
        #Scale using scalers for the highest key combination in the hierarchy
        
        if self.scaling_method == 'mean_scaling':
            # the global scaler is used for scaling keys which do not have an observation within the training period
            # high quantiles 0.8, 0.9 etc. can also be a good substitute
            npd_scale = np.maximum(df[df[self.time_index_col] <= self.train_till][self.target_col].quantile(0.9),
                                   df[df[self.time_index_col] <= self.train_till][self.target_col].mean())
            highest_key_cols = list(self.highest_key_combination)
            df['scaler'] = df[df[self.time_index_col] <= self.train_till].groupby(highest_key_cols)[self.target_col].transform(lambda x: np.maximum(x.mean()+1, 1.0))
            df['scaler'] = df.groupby(highest_key_cols)['scaler'].transform(lambda x: x.ffill().bfill().fillna(npd_scale))
            df[self.target_col] = df[self.target_col]/df['scaler']

        elif self.scaling_method == 'no_scaling':
            df['scaler'] = 1.0
            df[self.target_col] = df[self.target_col] / df['scaler']

        return df

    def scale_covariates(self, df):
        #Individually scale each 'id' & concatenate them all in one dataframe. Uses Joblib for parallelization.
        
        groups = df.groupby([self.id_col])
        scaled_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE, backend=backend, timeout=timeout)(delayed(self.df_scaler)(gdf) for _, gdf in groups)
        gdf = pd.concat(scaled_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        get_reusable_executor().shutdown(wait=True)

        return gdf

    def df_scaler(self, gdf):
        #Scale co-variates for lowest_key_level only

        # obtain scalers
        
        scale_gdf = gdf.reset_index(drop=True)

        if scale_gdf['key_level'].unique().tolist()[0] == self.covar_key_level:

            # for lowest level keys, scale both target & co-variates
            if self.scaling_method == 'mean_scaling':

                if len(self.temporal_known_num_col_list) > 0:
                    known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                    known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
                    known_scale = np.divide(known_sum, known_nz_count) + 1.0
                    # use max scale for known co-variates
                    #known_scale = np.maximum(np.nanmax(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                else:
                    known_scale = 1.0

            elif self.scaling_method == 'no_scaling':
                if len(self.temporal_known_num_col_list) > 0:
                    # use max scale for known co-variates
                    known_scale = np.maximum(np.nanmax(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                else:
                    known_scale = 1

            # reset index
            gdf = gdf.reset_index(drop=True)

            # scale each feature independently
            if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
                gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list]/known_scale

        return gdf
    """

    def scale_dataset(self, df):
        """
        Individually scale each 'id' & concatenate them all in one dataframe. Uses Joblib for parallelization.
        """
        # filter out ids with insufficient timestamps (at least one datapoint should be before train cutoff period)
        # df = df.groupby(self.id_col).filter(lambda x: x[self.time_index_col].min() < self.train_till)

        groups = df.groupby([self.id_col])
        scaled_gdfs = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE,
                               backend=backend, timeout=timeout)(delayed(self.df_scaler)(gdf) for _, gdf in groups)
        gdf = pd.concat(scaled_gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        get_reusable_executor().shutdown(wait=True)
        return gdf

    def df_scaler(self, gdf):
        """
        Scales a dataframe based on the chosen scaling method & columns specification
        """
        # obtain scalers
        if self.train_till <= gdf[self.time_index_col].min():
            # this handles edge cases where very few series are too short; these series may not get as good a result as
            # others due to poor generalization, but they won't be excluded from the forecast set either.
            scale_gdf = gdf[gdf[self.time_index_col] <= self.test_till].reset_index(drop=True)
        else:
            scale_gdf = gdf[gdf[self.time_index_col] <= self.train_till].reset_index(drop=True)
        covar_gdf = gdf.reset_index(drop=True)

        if self.scaling_method == 'mean_scaling':
            target_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.target_col])), 1.0)
            target_sum = np.sum(np.abs(scale_gdf[self.target_col]))
            scale = np.divide(target_sum, target_nz_count) + 1.0

            if len(self.temporal_known_num_col_list) > 0:
                known_nz_count = np.maximum(
                    np.count_nonzero(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                known_sum = np.sum(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0)
                known_scale = np.divide(known_sum, known_nz_count) + 1.0
                """
                # use max scale for known co-variates
                #known_scale = np.maximum(np.max(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                """
            else:
                known_scale = 1

            if len(self.temporal_unknown_num_col_list) > 0:
                # use max scale for known co-variates
                unknown_scale = np.maximum(np.max(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0),
                                           1.0)
            else:
                unknown_scale = 1

        elif self.scaling_method == 'standard_scaling':
            scale_mu = scale_gdf[self.target_col].mean()
            scale_std = np.maximum(scale_gdf[self.target_col].std(), 0.0001)
            scale = [scale_mu, scale_std]

            if len(self.temporal_known_num_col_list) > 0:
                known_nz_count = np.maximum(
                    np.count_nonzero(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                known_sum = np.sum(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0)
                known_scale = np.divide(known_sum, known_nz_count) + 1.0
                """
                # use max scale for known co-variates
                known_scale = np.maximum(np.max(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                """
            else:
                known_scale = 1

            if len(self.temporal_unknown_num_col_list) > 0:
                # use max scale for known co-variates
                unknown_scale = np.maximum(np.max(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0),
                                           1.0)
            else:
                unknown_scale = 1

        # new in 1.2
        elif self.scaling_method == 'quantile_scaling':
            med = scale_gdf[self.target_col].quantile(q=0.5)
            iqr = scale_gdf[self.target_col].quantile(q=self.iqr_high) - scale_gdf[self.target_col].quantile(q=self.iqr_low)
            scale = [med, np.maximum(iqr, 1.0)]

            if len(self.temporal_known_num_col_list) > 0:
                known_nz_count = np.maximum(
                    np.count_nonzero(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                known_sum = np.sum(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0)
                known_scale = np.divide(known_sum, known_nz_count) + 1.0
                """
                # use max scale for known co-variates
                known_scale = np.maximum(np.max(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                """
            else:
                known_scale = 1

            if len(self.temporal_unknown_num_col_list) > 0:
                # use max scale for known co-variates
                unknown_scale = np.maximum(np.max(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0),
                                           1.0)
            else:
                unknown_scale = 1

        elif self.scaling_method == 'no_scaling':
            scale = 1.0
            if len(self.temporal_known_num_col_list) > 0:
                # use max scale for known co-variates
                known_scale = np.maximum(np.max(np.abs(covar_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
            else:
                known_scale = 1

            if len(self.temporal_unknown_num_col_list) > 0:
                # use max scale for known co-variates
                unknown_scale = np.maximum(np.max(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0),
                                           1.0)
            else:
                unknown_scale = 1

        # reset index
        gdf = gdf.reset_index(drop=True)

        # scale each feature independently
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            gdf[self.target_col] = gdf[self.target_col] / scale
            gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list] / known_scale
            gdf[self.temporal_unknown_num_col_list] = gdf[self.temporal_unknown_num_col_list] / unknown_scale

        elif self.scaling_method == 'quantile_scaling':
            gdf[self.target_col] = (gdf[self.target_col] - scale[0]) / scale[1]
            gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list] / known_scale
            gdf[self.temporal_unknown_num_col_list] = gdf[self.temporal_unknown_num_col_list] / unknown_scale

        elif self.scaling_method == 'standard_scaling':
            gdf[self.target_col] = (gdf[self.target_col] - scale[0]) / scale[1]
            gdf[self.temporal_known_num_col_list] = gdf[self.temporal_known_num_col_list] / known_scale
            gdf[self.temporal_unknown_num_col_list] = gdf[self.temporal_unknown_num_col_list] / unknown_scale

        # limits for o/p clipping
        output_upper_limit = 1.5 * scale_gdf[self.target_col].max()
        output_lower_limit = 0.5 * scale_gdf[self.target_col].min()
        gdf['output_upper_limit'] = output_upper_limit
        gdf['output_lower_limit'] = output_lower_limit

        # Store scaler as a column
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            gdf['scaler'] = scale
        elif self.scaling_method == 'quantile_scaling':
            gdf['scaler_median'] = scale[0]
            gdf['scaler_iqr'] = scale[1]
        elif self.scaling_method == 'standard_scaling':
            gdf['scaler_mu'] = scale[0]
            gdf['scaler_std'] = scale[1]

        return gdf

    def log1p_transform_target(self, df):

        if self.log1p_transform:
            """
            For Tweedie loss, log1p transform is taken after scaling.
            Log1p transformation itself is optional.
            """
            print("   log1p transforming target ...")
            df[self.target_col] = np.log1p(df[self.target_col])
        else:
            pass

        return df

    def apply_agg_power_correction(self, df):
        """
        Applies power correction of p=0 for aggregate time-series in hierarchy as they are typically log-normally distributed

        """
        df['tweedie_p'] = np.where(df['key_level'] == self.covar_key_level, df['tweedie_p'], 0)

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
        
        if len(null_cols) > 0:
            null_status = True
        else:
            null_status = False
        
        return null_status, null_cols

    def onehot_encode(self, df):
        
        onehot_col_list = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list + \
                          self.global_context_col_list
        df = pd.concat([df[onehot_col_list], pd.get_dummies(data=df, columns=onehot_col_list, prefix_sep='_')],
                       axis=1, join='inner')
        
        return df

    def create_lead_lag_features(self, df):

        for col in [self.target_col] + \
                   self.temporal_known_num_col_list + \
                   self.temporal_unknown_num_col_list + \
                   self.known_onehot_cols + \
                   self.unknown_onehot_cols + \
                   self.rolling_feature_cols:

            # instantiate with empty lists
            self.lead_lag_features_dict[col] = []

            if col == self.target_col:
                for lag in range(self.max_target_lags, self.lag_offset, -1):
                    df[f'{col}_lag_{lag}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=lag, fill_value=0)
                    self.lead_lag_features_dict[col].append(f'{col}_lag_{lag}')
                for lead in range(0, self.fh):
                    df[f'{col}_fh_{lead}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=-lead, fill_value=0)
                    self.multistep_target.append(f'{col}_fh_{lead}')
            elif col == 'y_mask':
                for lead in range(0, self.fh):
                    df[f'{col}_fh_{lead}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=-lead)
                    self.multistep_mask.append(f'{col}_fh_{lead}')
            else:
                for lag in range(self.max_covar_lags, 0, -1):
                    df[f'{col}_lag_{lag}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=lag, fill_value=0)
                    self.lead_lag_features_dict[col].append(f'{col}_lag_{lag}')

            if col in self.temporal_known_num_col_list + self.known_onehot_cols:
                for lead in range(0, self.max_leads):
                    df[f'{col}_lead_{lead}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=-lead, fill_value=0)
                    self.lead_lag_features_dict[col].append(f'{col}_lead_{lead}')

            self.node_features_label[col] = self.lead_lag_features_dict[col]

        # don't drop rows with NaNs in lag/lead cols
        self.all_lead_lag_cols = list(itertools.chain.from_iterable([feat_col_list for col, feat_col_list in
                                                                     self.lead_lag_features_dict.items()]))

        return df

    def derive_rolling_features(self, df):
        """"
        Can be run once
        """
        self.rolling_feature_cols = []

        if len(self.rolling_features_list) > 0:
            for tup in self.rolling_features_list:
                if len(tup) >= 6:
                    raise ValueError("rolling feature tuples not defined properly.")
                else:
                    col = tup[0]
                    stat = tup[1]
                    window_size = tup[2]
                    offset = tup[3]
                    if len(tup) == 5:
                        parameter = tup[4]
                    # check
                    if col not in self.col_list:
                        raise ValueError("rolling feature window col not in columns list.")
                    if stat not in ['mean', 'quantile', 'trend_disruption', 'std']:
                        raise ValueError("stat not one of ['mean','quantile','trend_disruption','std'].")
                    if col != self.time_index_col:
                        feat_name = f'rolling_{stat}_by_{col}_win_{window_size}_offset_{offset}'
                        if stat == 'mean':
                            df[feat_name] = df.groupby([self.id_col, col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').mean())
                        elif stat == 'quantile':
                            df[feat_name] = df.groupby([self.id_col, col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').quantile(parameter))
                        elif stat == 'std':
                            df[feat_name] = df.groupby([self.id_col, col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').std().fillna(0))
                        self.rolling_feature_cols.append(feat_name)
                    else:
                        feat_name = f'rolling_{stat}_win_{window_size}_offset_{offset}'
                        if stat == 'mean':
                            df[feat_name] = df.groupby([self.id_col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').mean())
                        elif stat == 'quantile':
                            df[feat_name] = df.groupby([self.id_col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').quantile(parameter))
                        elif stat == 'std':
                            df[feat_name] = df.groupby([self.id_col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').std().fillna(0))
                        elif stat == 'trend_disruption':
                            # mv avg
                            df[feat_name + '_mvavg'] = df.groupby([self.id_col])[self.target_col].transform(
                                lambda x: x.shift(periods=offset).rolling(window_size, min_periods=1,
                                                                          closed='right').mean())
                            # actual/mvavg ratio
                            df[feat_name + '_r1'] = df[self.target_col] / df[feat_name + '_mvavg']
                            # trend disruption ratio
                            df[feat_name + '_r2'] = df[self.target_col] / df.groupby([self.id_col])[
                                self.target_col].shift(1)
                            # identify only disruption points
                            df[feat_name] = np.where((df[feat_name + '_r2'] >= 1.5) | (df[feat_name + '_r2'] <= 0.5), 1,
                                                     0)

                        self.rolling_feature_cols.append(feat_name)
        return df

    def pad_dataframe(self, df, dateindex):
        # this ensures num nodes in a graph don't change from period to period. Essentially, we introduce dummy nodes.
        
        # function to fill NaNs in group id & stat cols post padding
        def fillgrpid(x):
            id_val = x[self.id_col].unique().tolist()[0]
            x = dateindex.merge(x, on=[self.time_index_col], how='left').fillna({self.id_col: id_val})
            
            for col in self.global_context_col_list + self.global_context_onehot_cols + self.scaler_cols + \
                       self.tweedie_p_col + ['key_level', 'key_list', 'Key_Level_Weight', 'Key_Weight']:
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

        # check data sufficiency
        df = self.check_data_sufficiency(data)

        # create new keys
        print("   preprocessing dataframe - creating aggregate keys...")
        df = self.create_new_keys(df)

        # create new targets
        print("   preprocessing dataframe - creating new targets for aggregate keys...")
        df = self.create_new_targets(df)

        # create keybom
        print("   preprocessing dataframe - creating key bom...")
        if self.key_hierarchy is None:
            df_keybom = self.get_keybom(df)
        else:
            df_keybom = self.get_keybom_hierarchy(df)

        # stack subkey level dfs into one df
        print("   preprocessing dataframe - consolidating all keys into one df...")
        df = self.stack_key_level_dataframes(df, df_keybom)

        # del keybom
        del df_keybom
        gc.collect()

        # sort
        print("   preprocessing dataframe - sort by datetime & id...")
        df = self.sort_dataset(df)

        if self.log1p_transform:
            # estimate tweedie p
            if self.estimate_tweedie_p:
                print("   estimating tweedie p using GLM ...")
                df = self.parallel_tweedie_p_estimate(df)
                # apply power correction if required
                # print("   applying tweedie p correction for continuous ts, if applicable ...")
                # df = self.apply_agg_power_correction(df)
            # scale dataset
            print("   preprocessing dataframe - scale numeric cols...")
            df = self.scale_dataset(df)
            # apply log1p transform
            df = self.log1p_transform_target(df)
        else:
            # estimate tweedie p
            if self.estimate_tweedie_p:
                print("   estimating tweedie p using GLM ...")
                df = self.parallel_tweedie_p_estimate(df)
                # apply power correction if required
                # print("   applying tweedie p correction for continuous ts, if applicable ...")
                # df = self.apply_agg_power_correction(df)
            print("   preprocessing dataframe - scale numeric cols...")
            df = self.scale_dataset(df)

        # onehot encode
        print("   preprocessing dataframe - onehot encode categorical columns...")
        df = self.onehot_encode(df)

        print("   preprocessing dataframe - gather node specific feature cols...")
        # get onehot features
        for node in self.node_cols:
            if node in self.global_context_col_list:
                # one-hot col names
                onehot_cols_prefix = str(node) + '_'
                onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
                self.global_context_onehot_cols += onehot_col_features
            elif node in self.temporal_known_cat_col_list:
                # one-hot col names
                onehot_cols_prefix = str(node) + '_'
                onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
                self.known_onehot_cols += onehot_col_features
            elif node in self.temporal_unknown_cat_col_list:
                # one-hot col names
                onehot_cols_prefix = str(node) + '_'
                onehot_col_features = [col for col in df.columns.tolist() if col.startswith(onehot_cols_prefix)]
                self.unknown_onehot_cols += onehot_col_features

        print("\npreprocessed known_onehot_cols: ", self.known_onehot_cols)
        print("\npreprocessed unknown_onehot_cols: ", self.unknown_onehot_cols)
        print("\npreprocessed global_context_onehot_cols: ", self.global_context_onehot_cols)
        print("\npreprocessed temporal_known_num_col_list: ", self.temporal_known_num_col_list)
        print("\npreprocessed temporal_unknown_num_col_list: ", self.temporal_unknown_num_col_list)
        print("\nTotal Keys across hierarchy: ", df[self.id_col].nunique())
        for k in df['key_level'].unique().tolist():
            print("Total Keys for level {}: {}".format(k, df[df['key_level'] == k][self.id_col].nunique()))

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
        keybom_nested = torch.nested.nested_tensor(list(df_snap['mapped_key_list_arr'].values), dtype=torch.int16, requires_grad=False)
        keybom_padded = torch.nested.to_padded_tensor(keybom_nested, -1)

        # Create HeteroData Object
        data = HeteroData({"y_mask": None, "y_weight": None, "y_level_weight": None})
        
        # get node features
        data[self.target_col].x = torch.tensor(df_snap[self.lead_lag_features_dict[self.target_col]].to_numpy(), dtype=torch.float)
        data[self.target_col].y = torch.tensor(df_snap[self.multistep_target].to_numpy().reshape(-1, self.fh), dtype=torch.float)
        data[self.target_col].y_weight = torch.tensor(df_snap['Key_Weight'].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_level_weight = torch.tensor(df_snap['Key_Level_Weight'].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_mask = torch.tensor(df_snap[self.multistep_mask].to_numpy().reshape(-1, self.fh), dtype=torch.float)

        # in case target lags are not to be used as a feature
        if not self.autoregressive_target:
            data[self.target_col].x = torch.zeros_like(data[self.target_col].x)

        if len(self.scaler_cols) == 1:
            data[self.target_col].scaler = torch.tensor(df_snap['scaler'].to_numpy().reshape(-1, 1), dtype=torch.float)
            data['scaler'].x = torch.tensor(df_snap['scaler'].to_numpy().reshape(-1, 1), dtype=torch.float)
        else:
            data[self.target_col].scaler = torch.tensor(df_snap[self.scaler_cols].to_numpy().reshape(-1, 2), dtype=torch.float)
            data['scaler'].x = torch.tensor(df_snap[self.scaler_cols].to_numpy().reshape(-1, 2), dtype=torch.float)

        # applies only to tweedie
        if self.estimate_tweedie_p:
            data[self.target_col].tvp = torch.tensor(df_snap['tweedie_p'].to_numpy().reshape(-1, 1), dtype=torch.float)

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
            data[col].x = torch.tensor(df_snap[onehot_col_features].to_numpy(), dtype=torch.float)
                
        # directed edges between global context node & target_col nodes
        for col in self.global_context_col_list:
            col_unique_values = sorted(df_snap[col].unique().tolist())
            
            edges_stack = []
            for value in col_unique_values:
                # get subset of all nodes with common col value
                edges = df_snap[(df_snap[col] == value) & (df_snap['key_level'] == self.covar_key_level)][[col, self.id_col]].to_numpy()
                edges_stack.append(edges)

            # reverse edges
            edges = np.concatenate(edges_stack, axis=0)
            edge_name = (col, '{}_context'.format(col), self.target_col)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            
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
                edge_name = (self.target_col, 'related_by_{}_at_{}'.format(col, key_level), self.target_col)
                rev_edge_name = (self.target_col, 'rev_related_by_{}_at_{}'.format(col, key_level), self.target_col)
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
                edge_name = (self.target_col, 'aggregated_by_{}_minus_{}'.format(col, key_level), self.target_col)
                rev_edge_name = (self.target_col, 'rev_aggregated_by_{}_minus_{}'.format(col, key_level), self.target_col)
                # add edges to Data()
                edges = np.concatenate(fwd_edges_stack, axis=0)
                rev_edges = np.concatenate(rev_edges_stack, axis=0)
                data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
                data[rev_edge_name].edge_index = torch.tensor(rev_edges.transpose(), dtype=torch.long)

        # directed edges are from co-variates to target
        
        for col in self.temporal_known_num_col_list+self.temporal_unknown_num_col_list+self.known_onehot_cols+self.unknown_onehot_cols:

            nodes = df_snap[df_snap['key_level'] == self.covar_key_level][self.id_col].to_numpy()
            edges = np.column_stack([nodes, nodes])
                
            edge_name = (col, '{}_effects'.format(col), self.target_col)
            data[edge_name].edge_index = torch.tensor(edges.transpose(), dtype=torch.long)

        # validate dataset
        #print("validate snapshot graph ...")
        #data.validate(raise_on_error=True)

        # get memory consumption
        #target_tensor_size = (data[self.target_col].y.element_size() * data[self.target_col].y.nelement())
        #input_tensor_size = (data[self.target_col].x.element_size() * data[self.target_col].x.nelement())
        #mask_tensor_size = (data[self.target_col].y_mask.element_size() * data[self.target_col].y_mask.nelement())
        #weight_tensor_size = (data[self.target_col].y_weight.element_size() * data[self.target_col].y_weight.nelement())
        #keybom_tensor_size = (data['keybom'].x.element_size() * data['keybom'].x.nelement())
        #scaler_tensor_size = (data['scaler'].x.element_size() * data['scaler'].x.nelement())
        #key_level_tensor_size = (data[self.target_col].y_level_weight.element_size() * data[self.target_col].y_level_weight.nelement())

        #print("total graph attribute memory usage: ", target_tensor_size+input_tensor_size+mask_tensor_size+weight_tensor_size+key_level_tensor_size+keybom_tensor_size+scaler_tensor_size)
        #print("keybom tensor size: ", keybom_tensor_size)
        #print("keybom shape: ", data['keybom'].x.shape)

        return data

    def onetime_dataprep(self, df):
        # preprocess
        print("preprocessing dataframe...")
        df = self.preprocess(df)

        # pad dataframe if required (will return df unchanged if not)
        print("padding dataframe...")
        df = self.parallel_pad_dataframe(df)  # self.pad_dataframe(df)
        print("reduce mem usage post padding ...")
        df = self.reduce_mem_usage(df)
        print("creating relative time index & recency weights...")
        self.onetime_prep_df = self.get_relative_time_index(df)
        # add 'relative time index' to self.temporal_known_num_col_list
        self.temporal_known_num_col_list = self.temporal_known_num_col_list + ['relative_time_index']

    def create_train_test_dataset(self, df):

        print("create rolling features...")
        df = self.derive_rolling_features(df)
        self.temporal_unknown_num_col_list = self.temporal_unknown_num_col_list + self.rolling_feature_cols
        print("   new preprocessed temporal_unknown_num_col_list: ", self.temporal_unknown_num_col_list)
        # create lagged features
        print("create lead & lag features...")
        df = self.create_lead_lag_features(df)
        print("lead_lag_features_dict: ", self.lead_lag_features_dict)

        def parallel_snapshot_graphs(df, period):
            df_snap = df[df[self.time_index_col] == period].reset_index(drop=True)
            snapshot_graph = self.create_snapshot_graph(df_snap, period)
            return snapshot_graph

        print("create train/test cutoffs ...")
        train_cutoff, test_cutoff = self.split_train_test(df)

        # get train/test snapshots list
        snap_periods_list = sorted(df[df[self.time_index_col] <= test_cutoff][self.time_index_col].unique(), reverse=False)

        # restrict samples for very large datasets based on interleaving
        snap_periods_list = snap_periods_list[int(self.max_target_lags - 1):]

        if self.interleave > 1:
            snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]

        print("snap_periods_list: {}".format(snap_periods_list))

        # for each split create graph dataset iterator
        print("gather snapshot graphs...")

        if self.subgraph_sample_col is not None:
            all_subgraph_col_values = df[self.subgraph_sample_col].unique().tolist()
            # shuffle
            random.shuffle(all_subgraph_col_values)
            # sample
            snapshot_list = []
            if self.subgraph_sample_size > 0:
                train_snapshots = []
                test_snapshots = []
                for i in range(0, len(all_subgraph_col_values), int(self.subgraph_sample_size)):
                    df_sample = df[df[self.subgraph_sample_col].isin(all_subgraph_col_values[i:i + self.subgraph_sample_size])]
                    # sample snapshot graphs
                    print("  gathering for subgraph_col_values: ", all_subgraph_col_values[i:i + self.subgraph_sample_size])
                    sample_snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df_sample, period) for period in snap_periods_list)

                    # split into train/test
                    num_total_snapshots = len(sample_snapshot_list)
                    print("total samples picked: {}".format(num_total_snapshots))

                    num_recent_val_snapshots = int(num_total_snapshots * self.test_recent_percentage)
                    num_random_val_snapshots = int(num_total_snapshots * self.test_random_percentage)
                    num_nonrecent_snapshots = int(num_total_snapshots - num_recent_val_snapshots)
                    val_random_select_indices = random.sample(range(num_nonrecent_snapshots), num_random_val_snapshots)

                    recent_val_snapshots = sample_snapshot_list[-num_recent_val_snapshots:]
                    nonrecent_val_snapshots = [sample_snapshot_list[i] for i in val_random_select_indices]
                    test_snapshots += recent_val_snapshots + nonrecent_val_snapshots
                    train_snapshots += [sample_snapshot_list[i] for i in range(num_nonrecent_snapshots) if i not in val_random_select_indices]

                    del sample_snapshot_list
                    gc.collect()

                print("train samples picked: {}".format(len(train_snapshots)))
                print("test samples picked: {}".format(len(test_snapshots)))
                train_dataset = DataLoader(train_snapshots, batch_size=self.batch, shuffle=self.shuffle)
                test_dataset = DataLoader(test_snapshots, batch_size=self.batch, shuffle=self.shuffle)

            else:
                # sample snapshot graphs
                sample_snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)
                snapshot_list.append(sample_snapshot_list)

                # Create dataset iterators
                num_total_snapshots = len(list(itertools.chain.from_iterable(snapshot_list)))
                print("total samples picked: {}".format(num_total_snapshots))

                # split into train/test
                # test = last 10% of snapshots + randomly selected 10% of rest of the snapshots
                num_recent_val_snapshots = int(num_total_snapshots * self.test_recent_percentage)
                num_random_val_snapshots = int(num_total_snapshots * self.test_random_percentage)
                num_nonrecent_snapshots = int(num_total_snapshots - num_recent_val_snapshots)
                val_random_select_indices = random.sample(range(num_nonrecent_snapshots), num_random_val_snapshots)

                recent_val_snapshots = list(itertools.chain.from_iterable(snapshot_list))[-num_recent_val_snapshots:]
                nonrecent_val_snapshots = [list(itertools.chain.from_iterable(snapshot_list))[i] for i in val_random_select_indices]
                test_snapshots = recent_val_snapshots + nonrecent_val_snapshots
                train_snapshots = [list(itertools.chain.from_iterable(snapshot_list))[i] for i in range(num_nonrecent_snapshots) if i not in val_random_select_indices]

                print("train samples picked: {}".format(len(train_snapshots)))
                print("test samples picked: {}".format(len(test_snapshots)))
                train_dataset = DataLoader(train_snapshots, batch_size=self.batch, shuffle=self.shuffle)
                test_dataset = DataLoader(test_snapshots, batch_size=self.batch, shuffle=self.shuffle)

                del snapshot_list
                gc.collect()

        else:
            snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)

            # Create dataset iterators
            num_total_snapshots = len(snapshot_list)
            print("total samples picked: {}".format(num_total_snapshots))

            # split into train/test
            # test = last 10% of snapshots + randomly selected 10% of rest of the snapshots
            num_recent_val_snapshots = int(num_total_snapshots * self.test_recent_percentage)
            num_random_val_snapshots = int(num_total_snapshots * self.test_random_percentage)
            num_nonrecent_snapshots = int(num_total_snapshots - num_recent_val_snapshots)
            val_random_select_indices = random.sample(range(num_nonrecent_snapshots), num_random_val_snapshots)

            recent_val_snapshots = snapshot_list[-num_recent_val_snapshots:]
            nonrecent_val_snapshots = [snapshot_list[i] for i in val_random_select_indices]
            test_snapshots = recent_val_snapshots + nonrecent_val_snapshots
            train_snapshots = [snapshot_list[i] for i in range(num_nonrecent_snapshots) if i not in val_random_select_indices]

            print("train samples picked: {}".format(len(train_snapshots)))
            print("test samples picked: {}".format(len(test_snapshots)))

            train_dataset = DataLoader(train_snapshots, batch_size=self.batch, shuffle=self.shuffle)
            test_dataset = DataLoader(test_snapshots, batch_size=self.batch, shuffle=self.shuffle)

            del snapshot_list
            gc.collect()

        return train_dataset, test_dataset

    def create_train_test_dataset_orig(self, df):

        print("create rolling features...")
        df = self.derive_rolling_features(df)
        self.temporal_unknown_num_col_list = self.temporal_unknown_num_col_list + self.rolling_feature_cols
        print("   new preprocessed temporal_unknown_num_col_list: ", self.temporal_unknown_num_col_list)
        # create lagged features
        print("create lead & lag features...")
        df = self.create_lead_lag_features(df)

        # split into train,test,infer
        print("get cutoffs for training & testing periods ...")
        train_cutoff, test_cutoff = self.split_train_test(df)

        print("splitting dataframe for training & testing ...")
        train_df = df[df[self.time_index_col] <= train_cutoff]
        test_df = df[(df[self.time_index_col] > self.train_till) & (df[self.time_index_col] <= test_cutoff)]
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
                    snap_periods_list = snap_periods_list[int(self.max_target_lags - 1):]
                if (self.interleave > 1) and (df_type == 'train'):
                    snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]

                print("snap_periods_list: {}".format(snap_periods_list))

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

            else:
                # all snapshot timestamps
                snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)

                # restrict samples for very large datasets based on interleaving
                if df_type == 'train':
                    snap_periods_list = snap_periods_list[int(self.max_target_lags - 1):]
                if (self.interleave > 1) and (df_type == 'train'):
                    snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]

                print("snap_periods_list: {}".format(snap_periods_list))
                print("picking {} samples for {}".format(len(snap_periods_list), df_type))

                # sample snapshot graphs
                snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)

                dataset = DataLoader(snapshot_list, batch_size=self.batch, shuffle=self.shuffle)

                # append
                datasets[df_type] = dataset

        del df_dict
        gc.collect()
        train_dataset, test_dataset = datasets.get('train'), datasets.get('test')
        get_reusable_executor().shutdown(wait=True)

        return train_dataset, test_dataset

    def create_infer_dataset(self, df, infer_start):

        # drop lead/lag features if present
        try:
            df.drop(columns=self.all_lead_lag_cols+self.rolling_feature_cols, inplace=True)
            self.temporal_unknown_num_col_list = list(set(self.temporal_unknown_num_col_list) - set(self.rolling_feature_cols))
        except:
            pass

        print("create rolling features...")
        df = self.derive_rolling_features(df)
        self.temporal_unknown_num_col_list = self.temporal_unknown_num_col_list + self.rolling_feature_cols
        # create lagged features
        print("create lead & lag features...")
        df = self.create_lead_lag_features(df)

        infer_df = df[df[self.time_index_col] == infer_start]
        df_dict = {'infer': infer_df}
        
        # for each split create graph dataset iterator
        datasets = {}
        for df_type, df in df_dict.items():
            # snapshot start period: time.min() + max_history + fh, end_period:
            
            snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)[-1]

            # create individual snapshot graphs
            snapshot_list = []
            for period in [snap_periods_list]:
                df_snap = df[df[self.time_index_col] == period].reset_index(drop=True)
                snapshot_graph = self.create_snapshot_graph(df_snap, period)
                snapshot_list.append(snapshot_graph)
                # get node index map
                df_node_map_index = df[df[self.time_index_col] == period].reset_index(drop=True)
                self.node_index_map = self.node_indexing(df_node_map_index, [self.id_col])

            # Create a dataset iterator
            dataset = DataLoader(snapshot_list, batch_size=1, shuffle=False) 
            
            # append
            datasets[df_type] = dataset

        del df_dict
        gc.collect()
        infer_dataset = datasets.get('infer')

        return infer_df, infer_dataset

    def split_train_test(self, data):
        # multistep adjusted train/test cutoff
        train_cutoff = sorted(data[data[self.time_index_col] <= self.train_till][self.time_index_col].unique(), reverse=False)[-int(self.fh)]
        test_cutoff = sorted(data[data[self.time_index_col] <= self.test_till][self.time_index_col].unique(), reverse=False)[-int(self.fh)]

        print("train & test multistep cutoffs: ", train_cutoff, test_cutoff)
        return train_cutoff, test_cutoff

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

        infer_df = infer_df.groupby(self.id_col, sort=False).apply(lambda x: x[-1:]).reset_index(drop=True)
        print(infer_df[self.time_index_col].unique().tolist())

        infer_df = infer_df[[self.id_col, 'key_level', self.target_col, self.time_index_col] +
                            self.static_cat_col_list + self.global_context_col_list +
                            self.scaler_cols + self.tweedie_p_col]
        
        model_output = model_output.reshape(-1, self.fh)
        forecast_cols = [f'forecast_{h}' for h in range(self.fh)]
        output = pd.DataFrame(data=model_output, columns=forecast_cols)

        # merge forecasts with infer df
        output = pd.concat([infer_df, output], axis=1)    

        return output, forecast_cols
        
    def update_dataframe(self, df, output):
        
        # merge output & base_df
        reduced_output_df = output[[self.id_col, 'key_level', self.time_index_col, 'forecast']]
        df_updated = df.merge(reduced_output_df, on=[self.id_col, 'key_level', self.time_index_col], how='left')
        
        # update target for current ts with forecasts
        df_updated[self.target_col] = np.where(df_updated['forecast'].isnull(),
                                               df_updated[self.target_col],
                                               df_updated['forecast'])
        
        # drop forecast column
        df_updated = df_updated.drop(columns=['forecast'])
        
        return df_updated
    
    def build_dataset(self, df):
        # onetime prep
        self.onetime_dataprep(df)
        # build graph datasets for train/test
        if self.test_recent_percentage is None:
            self.train_dataset, self.test_dataset = self.create_train_test_dataset_orig(self.onetime_prep_df)
        else:
            self.train_dataset, self.test_dataset = self.create_train_test_dataset(self.onetime_prep_df)

    def build_infer_dataset(self, infer_start):
        # build graph datasets for infer
        try:
            del self.infer_dataset
            gc.collect()
        except:
            pass

        _, self.infer_dataset = self.create_infer_dataset(df=self.onetime_prep_df, infer_start=infer_start)

    def build(self,
              layer_type='HAN',
              model_dim=128,
              num_layers=1,
              num_rnn_layers=1,
              heads=1,
              forecast_quantiles=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9],
              dropout=0,
              skip_connection=False,
              device='cpu'):

        # key metadata for model def
        self.metadata = self.get_metadata(self.train_dataset)
        self.forecast_quantiles = forecast_quantiles
        sample_batch = next(iter(self.train_dataset))

        # target device to train on ['cuda','cpu']
        self.device = torch.device(device)

        # build model
        self.model = STGNN(hidden_channels=model_dim,
                           metadata=self.metadata,
                           target_node=self.target_col,
                           time_steps=self.fh,
                           n_quantiles=len(self.forecast_quantiles),
                           num_layers=num_layers,
                           num_rnn_layers=num_rnn_layers,
                           heads=heads,
                           dropout=dropout,
                           tweedie_out=self.tweedie_out,
                           skip_connection=skip_connection,
                           layer_type=layer_type)

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
            pytorch_total_params = sum(
                [0 if isinstance(p, torch.nn.parameter.UninitializedParameter) else p.numel() for p in
                 self.model.parameters()])
            print("total model params: ", pytorch_total_params)

    def train(self,
              lr,
              min_epochs,
              max_epochs,
              patience,
              min_delta,
              model_prefix,
              loss='Quantile',  # 'Tweedie','SMAPE','RMSE'
              delta=1.0,  # for Huber
              epsilon=0.01,  # for SMAPE
              use_amp=False,
              use_lr_scheduler=True,
              scheduler_params={'factor': 0.5, 'patience': 3, 'threshold': 0.0001, 'min_lr': 0.00001},
              sample_weights=False,
              stop_training_criteria='loss'):

        self.loss = loss

        if self.loss == 'Tweedie':
            # convert tweedie_variance_power to tensors
            self.tweedie_variance_power = [torch.tensor(i, dtype=torch.float32).reshape([1, 1]).to(self.device) for i in self.tweedie_variance_power]
            loss_fn = TweedieLoss(p_list=self.tweedie_variance_power)
        elif self.loss == 'Quantile':
            loss_fn = QuantileLoss(quantiles=self.forecast_quantiles)
        elif self.loss == 'SMAPE':
            loss_fn = SMAPE(epsilon=epsilon)
        elif self.loss == 'RMSE':
            loss_fn = RMSE()
        elif self.loss == 'Huber':
            loss_fn = Huber(delta=delta)

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
        # metrics
        train_rmse_hist = []
        val_rmse_hist = []

        # torch.amp -- for mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        def train_fn():
            self.model.train(True)
            total_examples = 0
            total_loss = 0
            total_mse = 0
            for i, batch in enumerate(self.train_dataset):

                if not self.grad_accum:
                    optimizer.zero_grad()

                batch = batch.to(self.device)
                batch_size = batch.num_graphs
                out = self.model(batch.x_dict, batch.edge_index_dict)

                if self.loss == 'Tweedie' and self.estimate_tweedie_p:
                    tvp = batch[self.target_col].tvp
                    tvp = torch.reshape(tvp, (-1, 1))
                else:
                    tvp = []

                # compute loss masking out N/A targets -- last snapshot
                if self.loss == 'Tweedie':
                    loss = loss_fn.loss(y_pred=out, y_true=batch[self.target_col].y, p=tvp,
                                        scaler=batch[self.target_col].scaler,
                                        log1p_transform=self.log1p_transform)
                else:
                    loss = loss_fn.loss(out, batch[self.target_col].y)

                mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)

                # key weight
                if sample_weights:
                    wt = torch.unsqueeze(batch[self.target_col].y_weight, dim=2)
                else:
                    wt = 1

                # key level weight
                key_level_wt = torch.unsqueeze(batch[self.target_col].y_level_weight, dim=2)

                # recency wt
                if self.recency_weights:
                    recency_wt = torch.unsqueeze(batch[self.target_col].recency_weight, dim=2)
                else:
                    recency_wt = 1

                weighted_loss = torch.mean(loss * mask * wt * key_level_wt * recency_wt)

                # metric
                if self.loss == 'Tweedie':
                    out = torch.exp(out) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                else:
                    out = out * torch.unsqueeze(batch[self.target_col].scaler, dim=2)

                actual = torch.unsqueeze(batch[self.target_col].y, dim=2) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                mse_err = (recency_wt * mask * wt * (out - actual) * (out - actual)).mean().data

                # normalize loss to account for batch accumulation
                if self.grad_accum:
                    weighted_loss = weighted_loss / self.accum_iter
                    weighted_loss.backward()
                    # weights update
                    if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_dataset)):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    weighted_loss.backward()
                    optimizer.step()

                total_examples += batch_size
                total_loss += float(weighted_loss)
                total_mse += mse_err

            return total_loss / total_examples, math.sqrt(total_mse / total_examples)

        def test_fn():
            self.model.train(False)
            total_examples = 0
            total_loss = 0
            total_mse = 0
            with torch.no_grad():
                for i, batch in enumerate(self.test_dataset):
                    batch_size = batch.num_graphs
                    batch = batch.to(self.device)
                    out = self.model(batch.x_dict, batch.edge_index_dict)

                    if self.loss == 'Tweedie' and self.estimate_tweedie_p:
                        tvp = batch[self.target_col].tvp
                        tvp = torch.reshape(tvp, (-1, 1))
                    else:
                        tvp = []

                    # compute loss masking out N/A targets -- last snapshot
                    if self.loss == 'Tweedie':
                        loss = loss_fn.loss(y_pred=out, y_true=batch[self.target_col].y, p=tvp,
                                            scaler=batch[self.target_col].scaler,
                                            log1p_transform=self.log1p_transform)
                    else:
                        loss = loss_fn.loss(out, batch[self.target_col].y)

                    mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)

                    if sample_weights:
                        wt = torch.unsqueeze(batch[self.target_col].y_weight, dim=2)
                    else:
                        wt = 1

                    # key level weight
                    key_level_wt = torch.unsqueeze(batch[self.target_col].y_level_weight, dim=2)

                    if self.recency_weights:
                        recency_wt = torch.unsqueeze(batch[self.target_col].recency_weight, dim=2)
                    else:
                        recency_wt = 1

                    weighted_loss = torch.mean(loss * mask * wt * key_level_wt * recency_wt)

                    # metric
                    if self.loss == 'Tweedie':
                        out = torch.exp(out) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                    else:
                        out = out * torch.unsqueeze(batch[self.target_col].scaler, dim=2)

                    actual = torch.unsqueeze(batch[self.target_col].y, dim=2) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                    mse_err = (recency_wt * mask * wt * (out - actual) * (out - actual)).mean().data

                    total_examples += batch_size
                    total_loss += float(weighted_loss)
                    total_mse += mse_err

            return total_loss / total_examples, math.sqrt(total_mse / total_examples)

        def train_amp_fn():
            self.model.train(True)
            total_examples = 0
            total_loss = 0
            total_mse = 0
            for i, batch in enumerate(self.train_dataset):

                if not self.grad_accum:
                    optimizer.zero_grad()

                batch = batch.to(self.device)
                batch_size = batch.num_graphs

                if self.loss == 'Tweedie' and self.estimate_tweedie_p:
                    tvp = batch[self.target_col].tvp
                    tvp = torch.reshape(tvp, (-1, 1))
                else:
                    tvp = []

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(batch.x_dict, batch.edge_index_dict)

                    # compute loss masking out N/A targets -- last snapshot
                    if self.loss == 'Tweedie':
                        loss = loss_fn.loss(y_pred=out, y_true=batch[self.target_col].y, p=tvp, scaler=batch[self.target_col].scaler, log1p_transform=self.log1p_transform)
                    else:
                        loss = loss_fn.loss(out, batch[self.target_col].y)

                    mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)

                    # key weight
                    if sample_weights:
                        wt = torch.unsqueeze(batch[self.target_col].y_weight, dim=2)
                    else:
                        wt = 1

                    # key level weight
                    key_level_wt = torch.unsqueeze(batch[self.target_col].y_level_weight, dim=2)

                    # recency wt
                    if self.recency_weights:
                        recency_wt = torch.unsqueeze(batch[self.target_col].recency_weight, dim=2)
                    else:
                        recency_wt = 1

                    weighted_loss = torch.mean(loss * mask * wt * key_level_wt * recency_wt)

                    # metric
                    if self.loss == 'Tweedie':
                        out = torch.exp(out) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                    else:
                        out = out * torch.unsqueeze(batch[self.target_col].scaler, dim=2)

                    actual = torch.unsqueeze(batch[self.target_col].y, dim=2) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                    mse_err = (recency_wt * mask * wt * (out - actual) * (out - actual)).mean().data

                # normalize loss to account for batch accumulation
                if self.grad_accum:
                    weighted_loss = weighted_loss / self.accum_iter
                    scaler.scale(weighted_loss).backward()
                    # weights update
                    if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_dataset)):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    scaler.scale(weighted_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                total_examples += batch_size
                total_loss += float(weighted_loss)
                total_mse += mse_err

            return total_loss / total_examples, math.sqrt(total_mse / total_examples)

        def test_amp_fn():
            self.model.train(False)
            total_examples = 0
            total_loss = 0
            total_mse = 0
            with torch.no_grad():
                for i, batch in enumerate(self.test_dataset):
                    batch_size = batch.num_graphs
                    batch = batch.to(self.device)

                    if self.loss == 'Tweedie' and self.estimate_tweedie_p:
                        tvp = batch[self.target_col].tvp
                        tvp = torch.reshape(tvp, (-1, 1))
                    else:
                        tvp = []

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(batch.x_dict, batch.edge_index_dict)

                        # compute loss masking out N/A targets -- last snapshot
                        if self.loss == 'Tweedie':
                            loss = loss_fn.loss(y_pred=out, y_true=batch[self.target_col].y, p=tvp, scaler=batch[self.target_col].scaler, log1p_transform=self.log1p_transform)
                        else:
                            loss = loss_fn.loss(out, batch[self.target_col].y)

                        mask = torch.unsqueeze(batch[self.target_col].y_mask, dim=2)

                        if sample_weights:
                            wt = torch.unsqueeze(batch[self.target_col].y_weight, dim=2)
                        else:
                            wt = 1

                        # key level weight
                        key_level_wt = torch.unsqueeze(batch[self.target_col].y_level_weight, dim=2)

                        if self.recency_weights:
                            recency_wt = torch.unsqueeze(batch[self.target_col].recency_weight, dim=2)
                        else:
                            recency_wt = 1

                        weighted_loss = torch.mean(loss * mask * wt * key_level_wt * recency_wt)
                        # metric
                        if self.loss == 'Tweedie':
                            out = torch.exp(out) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                        else:
                            out = out * torch.unsqueeze(batch[self.target_col].scaler, dim=2)

                        actual = torch.unsqueeze(batch[self.target_col].y, dim=2) * torch.unsqueeze(batch[self.target_col].scaler, dim=2)
                        mse_err = (recency_wt * mask * wt * (out - actual) * (out - actual)).mean().data

                    total_examples += batch_size
                    total_loss += float(weighted_loss)
                    total_mse += mse_err

            return total_loss / total_examples, math.sqrt(total_mse / total_examples)

        for epoch in range(max_epochs):

            if use_amp:
                loss, rmse = train_amp_fn()
                val_loss, val_rmse = test_amp_fn()
            else:
                loss, rmse = train_fn()
                val_loss, val_rmse = test_fn()

            print('EPOCH {}: Train loss: {}, Val loss: {}'.format(epoch, loss, val_loss))
            print('EPOCH {}: Train rmse: {}, Val rmse: {}'.format(epoch, rmse, val_rmse))

            if use_lr_scheduler:
                scheduler.step(val_loss)

            train_loss_hist.append(loss)
            val_loss_hist.append(val_loss)

            train_rmse_hist.append(rmse)
            val_rmse_hist.append(val_rmse)

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
            if len(model_list) > patience:
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

    def infer(self, infer_start, select_quantile):

        base_df = self.onetime_prep_df

        # print model used for inference
        print("running inference using best saved model: ", self.best_model)

        # infer fn
        def infer_fn(model, model_path, infer_data):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            output = []
            with torch.no_grad():
                for i, batch in enumerate(infer_data):
                    batch = batch.to(self.device)
                    out = model(batch.x_dict, batch.edge_index_dict)
                    output.append(out)
            return output

        print("forecasting starting period {}".format(infer_start))

        # infer dataset creation
        infer_df, infer_dataset = self.create_infer_dataset(base_df, infer_start=infer_start)
        output = infer_fn(self.model, self.best_model, infer_dataset)

        # select output quantile
        output_arr = output[0]
        output_arr = output_arr.cpu().numpy()

        # quantile selection
        min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

        assert min_qtile <= select_quantile <= max_qtile, "selected quantile out of bounds!"

        if self.loss == 'Tweedie':
            output_arr = output_arr[:, :, 0]
            output_arr = np.exp(output_arr)
        elif self.loss in ['RMSE', 'SMAPE', 'Huber']:
            output_arr = output_arr[:, :, 0]
        else:
            try:
                q_index = self.forecast_quantiles.index(select_quantile)
                output_arr = output_arr[:, :, q_index]
            except:
                q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                q_lower = int(q_upper - 1)
                q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower]) / (
                        self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                q_lower_weight = 1 - q_upper_weight
                output_arr = q_upper_weight * output_arr[:, :, q_upper] + q_lower_weight * output_arr[:, :, q_lower]

        # show current o/p
        forecast_df, forecast_cols = self.process_output(infer_df, output_arr)

        # reverse log1p transform after re-scaling
        if self.log1p_transform:
            for col in forecast_cols:
                forecast_df[col] = np.expm1(forecast_df[col])
            for col in self.multistep_targets:
                forecast_df[col] = np.expm1(forecast_df[col])

        # re-scale output
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler']
        elif self.scaling_method == 'quantile_scaling':
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
        else:
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_std'] + forecast_df['scaler_mu']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_std'] + forecast_df['scaler_mu']

        return forecast_df, forecast_cols

    def infer_sim(self, infer_start, select_quantile, sim_df):

        # print model used for inference
        print("running simulated inference using best saved model: ", self.best_model)

        # infer fn
        def infer_fn(model, model_path, infer_dataset):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.train(False)
            output = []
            with torch.no_grad():
                for _, batch in enumerate(infer_dataset):
                    batch = batch.to(self.device)
                    out = model(batch.x_dict, batch.edge_index_dict)
                    output.append(out)
            return output

        print("forecasting starting period {}".format(infer_start))

        # infer dataset creation
        infer_df, infer_dataset = self.create_infer_dataset(sim_df, infer_start=infer_start)
        output = infer_fn(self.model, self.best_model, infer_dataset)

        # select output quantile
        output_arr = output[0]
        output_arr = output_arr.cpu().numpy()

        # quantile selection
        min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

        assert min_qtile <= select_quantile <= max_qtile, "selected quantile out of bounds!"

        if self.loss == 'Tweedie':
            output_arr = output_arr[:, :, 0]
            output_arr = np.exp(output_arr)
        elif self.loss in ['RMSE', 'SMAPE', 'Huber']:
            output_arr = output_arr[:, :, 0]
        else:
            try:
                q_index = self.forecast_quantiles.index(select_quantile)
                output_arr = output_arr[:, :, q_index]
            except:
                q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                q_lower = int(q_upper - 1)
                q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower]) / (
                        self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                q_lower_weight = 1 - q_upper_weight
                output_arr = q_upper_weight * output_arr[:, :, q_upper] + q_lower_weight * output_arr[:, :, q_lower]

        # show current o/p
        forecast_df, forecast_cols = self.process_output(infer_df, output_arr)

        # reverse log1p transform after re-scaling
        if self.log1p_transform:
            for col in forecast_cols:
                forecast_df[col] = np.expm1(forecast_df[col])
            for col in self.multistep_targets:
                forecast_df[col] = np.expm1(forecast_df[col])

        # re-scale output
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler']
        elif self.scaling_method == 'quantile_scaling':
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
        else:
            for col in forecast_cols:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_std'] + forecast_df['scaler_mu']
            for col in self.multistep_targets:
                forecast_df[col] = forecast_df[col] * forecast_df['scaler_std'] + forecast_df['scaler_mu']

        return forecast_df, forecast_cols
