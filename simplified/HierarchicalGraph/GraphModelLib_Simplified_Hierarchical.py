#!/usr/bin/env python
# coding: utf-8
import random

# Model Specific imports
import torch
import copy
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, HeteroConv, SAGEConv, BatchNorm, LayerNorm
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
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
    def __init__(self):
        super().__init__()

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, p: torch.Tensor, scaler, log1p_transform):
        """
        if log1p_transform:
            # log1p first, scale next
            y_true = y_true * scaler
            y_pred = torch.squeeze(y_pred, dim=2)
            # reverse log of prediction y_pred
            #y_pred = torch.exp(y_pred)
            # rescale
            #y_pred = y_pred * scaler
            # get pred
            #y_pred = torch.expm1(y_pred)
            # take log of y_pred again
            #y_pred = torch.log(y_pred + 1e-8)

            a = y_true * torch.exp((y_pred + torch.log(scaler)) * (1 - p)) / (1 - p)
            b = torch.exp((y_pred + torch.log(scaler)) * (2 - p)) / (2 - p)
            loss = -a + b
        """
        # convert all 2-d inputs to 3-d
        y_true = torch.unsqueeze(y_true, dim=2)
        scaler = torch.unsqueeze(scaler, dim=2)
        p = torch.unsqueeze(p, dim=2)

        if log1p_transform:
            # scale first, log1p after
            y_true = torch.expm1(y_true) * scaler
            # reverse log of prediction y_pred
            y_pred = torch.exp(y_pred)
            # get pred
            y_pred = torch.expm1(y_pred)
            # take log of y_pred again
            y_pred = y_pred * scaler
            y_pred = torch.log(y_pred + 1e-8)

            a = y_true * torch.exp(y_pred * (1 - p)) / (1 - p)
            b = torch.exp(y_pred * (2 - p)) / (2 - p)
            loss = -a + b
        else:
            # no log1p
            y_true = y_true * scaler
            a = y_true * torch.exp((y_pred + torch.log(scaler)) * (1 - p)) / (1 - p)
            b = torch.exp((y_pred + torch.log(scaler)) * (2 - p)) / (2 - p)
            loss = -a + b

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
                conv_dict[e] = SAGEConv(in_channels=in_channels, out_channels=out_channels)
            else:
                if first_layer:
                    conv_dict[e] = SAGEConv(in_channels=in_channels, out_channels=out_channels, bias=False)
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
                x = norm(self.dropout(x_dict[node_type]).relu())
                x_dict[node_type] = x
        return x_dict


class HeteroSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, node_types, edge_types, is_output_layer=False):
        super().__init__()

        self.conv = HeteroConv({
            edge_type: SAGEConv(in_channels, out_channels) for edge_type in edge_types
        })

        if not is_output_layer:
            self.dropout = torch.nn.Dropout(dropout)
            self.norm_dict = torch.nn.ModuleDict({
                node_type:
                BatchNorm(out_channels)
                for node_type in node_types
            })

        self.is_output_layer = is_output_layer

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        print(x_dict)
        if not self.is_output_layer:
            for node_type, norm in self.norm_dict.items():
                print(node_type, x_dict[node_type])
                x = norm(self.dropout(x_dict[node_type]).relu())
                x_dict[node_type] = x
        return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, node_types, edge_types,
                 target_node_type, skip_connection=True):
        super().__init__()

        self.target_node_type = target_node_type
        self.skip_connection = skip_connection

        if num_layers == 1:
            self.skip_connection = False

        self.project_lin = Linear(hidden_channels, out_channels)

        # Transform/Feature Extraction Layers
        self.transformed_feat_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            if node_type == target_node_type:
                self.transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                                                                      hidden_size=hidden_channels,
                                                                      num_layers=1,
                                                                      batch_first=True)

        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            """
            conv = HeteroSAGEConv(
                in_channels=in_channels,  # if i == 0 else hidden_channels,
                out_channels=out_channels if i == num_layers - 1 else hidden_channels,
                dropout=dropout,
                node_types=node_types,
                edge_types=edge_types,
                is_output_layer=i == num_layers - 1,
            )
            """
            conv = HeteroForecastSageConv(in_channels=in_channels if i == 0 else hidden_channels,
                                          out_channels=hidden_channels, #out_channels if i == num_layers - 1 else hidden_channels,
                                          dropout=dropout,
                                          node_types=node_types,
                                          edge_types=edge_types,
                                          target_node_type=target_node_type,
                                          first_layer=i == 0,
                                          is_output_layer=i == num_layers - 1)

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        # transform target node
        for node_type, x in x_dict.items():
            if node_type == self.target_node_type:
                o, _ = self.transformed_feat_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3 -d (N,L,1)
                x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)

        if self.skip_connection:
            res_dict = x_dict

        # run convolutions
        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict)

            if self.skip_connection:
                res_dict = {key: res_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x) in zip(x_dict.items(), res_dict.items()) if
                          key == res_key}
                x_dict = {key: x.relu() for key, x in x_dict.items()}

        out = self.project_lin(x_dict[self.target_node_type])

        return out  #x_dict[self.target_node_type]


# Models

class STGNN(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_layers,
                 metadata,
                 target_node,
                 time_steps=1,
                 n_quantiles=1,
                 dropout=0.0,
                 tweedie_out=False,
                 skip_connection=True):

        super(STGNN, self).__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.time_steps = time_steps
        self.n_quantiles = n_quantiles
        self.tweedie_out = tweedie_out

        self.gnn_model = HeteroGraphSAGE(in_channels=(-1, -1),
                                         hidden_channels=hidden_channels,
                                         num_layers=num_layers,
                                         out_channels=int(n_quantiles * time_steps),
                                         dropout=dropout,
                                         node_types=self.node_types,
                                         edge_types=self.edge_types,
                                         target_node_type=target_node,
                                         skip_connection=skip_connection)

    def sum_over_index(self, x, x_index):
        return torch.index_select(x, 0, x_index).sum(dim=0)

    def log_transformed_sum_over_index(self, x, x_index):
        """
        the output is expected to be the log of required prediction, so, reverse log transform before aggregating.
        """
        return torch.exp(torch.index_select(x, 0, x_index)).sum(dim=0)

    def forward(self, x_dict, edge_index_dict):
        # get keybom
        keybom = x_dict['keybom']

        # get key_aggregation_status
        key_agg_status = x_dict['key_aggregation_status']
        agg_indices = (key_agg_status == 1).nonzero(as_tuple=True)[0].tolist()

        # del keybom from x_dict
        del x_dict['keybom']
        del x_dict['key_aggregation_status']

        # gnn model
        out = self.gnn_model(x_dict, edge_index_dict)

        out = torch.reshape(out, (-1, self.time_steps, self.n_quantiles))

        # fallback to this approach (slower) in case vmap doesn't work
        # constrain the higher level key o/ps to be the sum of their constituents
        """
        for i in agg_indices:
            out[i] = torch.index_select(out, 0, keybom[i][keybom[i] != -1]).sum(dim=0)
        """

        # vectorized approach follows:
        device_int = out.get_device()

        if device_int == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        dummy_out = torch.zeros(1, out.shape[1], out.shape[2]).to(device)
        # add a zero vector to the out tensor as workaround to the limitation of vmap of not being able to process
        # nested/dynamic shape tensors
        out = torch.cat([out, dummy_out], dim=0)

        # replace -1 from key bom with last dim in out
        keybom[keybom == -1] = int(out.shape[0] - 1)

        # call vmap on sum_over_index function
        if self.tweedie_out:
            batched_sum_over_index = torch.vmap(self.log_transformed_sum_over_index, in_dims=(None, 0), randomness='error')
            out = batched_sum_over_index(out, keybom)
            # again do the log_transform on the aggregates
            out = torch.log(out)
        else:
            batched_sum_over_index = torch.vmap(self.sum_over_index, in_dims=(None, 0), randomness='error')
            out = batched_sum_over_index(out, keybom)

        # returned shape of out should be same as that before cat with dummy_out

        return out


# Graph Object

class graphmodel():
    def __init__(self, 
                 col_dict,
                 max_target_lags,
                 max_covar_lags,
                 max_leads,
                 train_till,
                 test_till,
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
                 iqr_high=0.75,
                 iqr_low=0.25,
                 categorical_onehot_encoding=True,
                 directed_graph=True,
                 shuffle=True,
                 interleave=1,
                 recency_weights=False,
                 recency_alpha=0,
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
        
        """
        super().__init__()
        
        self.col_dict = copy.deepcopy(col_dict)
        self.min_history = int(min_history)
        self.fh = int(fh)
        self.max_history = int(1)
        self.max_target_lags = int(max_target_lags) if (max_target_lags is not None) and (max_target_lags > 0) else 1
        self.max_covar_lags = int(max_covar_lags) if (max_covar_lags is not None) and (max_covar_lags > 0) else 1
        self.max_leads = int(max_leads) if (max_leads is not None) and (max_leads > 0) else 1

        assert self.max_leads >= self.fh, "max_leads must be >= fh"
        
        # adjust train_till/test_till for delta|max_leads - fh| in split_* methods
        self.train_till = train_till
        self.test_till = test_till
        
        self.batch = batch
        self.grad_accum = grad_accum
        self.accum_iter = accum_iter
        self.scaling_method = scaling_method
        self.log1p_transform = log1p_transform
        self.tweedie_out = tweedie_out
        self.estimate_tweedie_p = estimate_tweedie_p
        self.tweedie_p_range = tweedie_p_range
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
        
        self.pad_constant = 0

        # extract column sets from col_dict
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

        # full column set for train/test/infer
        self.col_list = [self.id_col] + [self.target_col] + \
                         self.static_cat_col_list + self.global_context_col_list + \
                         self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                         self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.cat_col_list = self.global_context_col_list + self.static_cat_col_list + \
                            self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

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
        df['Key_Level_Weight'] = df['key_level'].map(self.key_levels_weight_dict)
        # Add 'Key_Weight' col
        if self.wt_col is None:
            df['Key_Weight'] = 1
        else:
            df['Key_Weight'] = np.where(df[self.wt_col].isnull(), 1, df[self.wt_col])
        # new col list
        self.col_list = df.columns.tolist()
        return df

    def check_data_sufficiency(self, df):
        """
        Exclude keys which do not have at least one data point within the training cutoff period
        """
        df = df.groupby(list(self.lowest_key_combination)).filter(lambda x: len(x[x[self.time_index_col] <= self.test_till]) >= self.min_history)

        return df

    def power_optimization_loop(self, df):
        # initialization constants
        init_power = 1.01
        max_iterations = 100

        try:
            endog = df[self.target_col].astype(np.float32).to_numpy()
            exog = df[self.temporal_known_num_col_list].astype(np.float32).to_numpy()

            # fit glm model
            def glm_fit(endog, exog, power):
                res = sm.GLM(endog, exog,
                             family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=power)).fit()
                return res.mu, res.scale, res._endog

            # optimize 1 iter
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
            df['tweedie_p'] = 1.50

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

    def scale_target(self, df):
        """
        Scale using scalers for the highest key combination in the hierarchy
        """
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
        Scale co-variates for lowest_key_level only
        """
        # obtain scalers
        
        scale_gdf = gdf.reset_index(drop=True)

        if scale_gdf['key_level'].unique().tolist()[0] == self.covar_key_level:

            # for lowest level keys, scale both target & co-variates
            if self.scaling_method == 'mean_scaling':

                if len(self.temporal_known_num_col_list) > 0:
                    """
                    known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
                    known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
                    known_scale = np.divide(known_sum, known_nz_count) + 1.0
                    """
                    # use max scale for known co-variates
                    known_scale = np.maximum(np.nanmax(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
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
                   self.unknown_onehot_cols:

            # instantiate with empty lists
            self.lead_lag_features_dict[col] = []

            if col == self.target_col:
                for lag in range(self.max_target_lags, 0, -1):
                    df[f'{col}_lag_{lag}'] = df.groupby(self.id_col, sort=False)[col].shift(periods=lag, fill_value=0)
                    self.lead_lag_features_dict[col].append(f'{col}_lag_{lag}')
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
        df_keybom = self.get_keybom(df)

        # stack subkey level dfs into one df
        print("   preprocessing dataframe - consolidating all keys into one df...")
        df = self.stack_key_level_dataframes(df, df_keybom)

        # sort
        print("   preprocessing dataframe - sort by datetime & id...")
        df = self.sort_dataset(df)

        if self.log1p_transform:
            # estimate tweedie p
            if self.estimate_tweedie_p:
                print("   estimating tweedie p using GLM ...")
                df = self.parallel_tweedie_p_estimate(df)
                # apply power correction if required
                print("   applying tweedie p correction for continuous ts, if applicable ...")
                df = self.apply_agg_power_correction(df)
            # scale dataset
            print("   preprocessing dataframe - scale target...")
            df = self.scale_target(df)
            print("   preprocessing dataframe - scale numeric known cols...")
            df = self.scale_covariates(df)
            # apply log1p transform
            df = self.log1p_transform_target(df)
        else:
            # scale dataset
            print("   preprocessing dataframe - scale target...")
            df = self.scale_target(df)
            print("   preprocessing dataframe - scale numeric known cols...")
            df = self.scale_covariates(df)
            # estimate tweedie p
            if self.estimate_tweedie_p:
                print("   estimating tweedie p using GLM ...")
                df = self.parallel_tweedie_p_estimate(df)
                # apply power correction if required
                print("   applying tweedie p correction for continuous ts, if applicable ...")
                df = self.apply_agg_power_correction(df)

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

        print("   preprocessed known_onehot_cols: ", self.known_onehot_cols)
        print("\n")
        print("   preprocessed unknown_onehot_cols: ", self.unknown_onehot_cols)
        print("\n")
        print("   preprocessed global_context_onehot_cols: ", self.global_context_onehot_cols)
        print("\n")
        print("   preprocessed temporal_known_num_col_list: ", self.temporal_known_num_col_list)
        print("\n")
        print("   preprocessed temporal_unknown_num_col_list: ", self.temporal_unknown_num_col_list)

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
        data = HeteroData({"y_mask": None, "y_weight": None, "y_level_weight": None})
        
        # get node features
        data[self.target_col].x = torch.tensor(df_snap[self.lead_lag_features_dict[self.target_col]].to_numpy(), dtype=torch.float)
        data[self.target_col].y = torch.tensor(df_snap[self.target_col].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_weight = torch.tensor(df_snap['Key_Weight'].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_level_weight = torch.tensor(df_snap['Key_Level_Weight'].to_numpy().reshape(-1, 1), dtype=torch.float)
        data[self.target_col].y_mask = torch.tensor(df_snap['y_mask'].to_numpy().reshape(-1, 1), dtype=torch.float)

        if len(self.scaler_cols) == 1:
            data[self.target_col].scaler = torch.tensor(df_snap['scaler'].to_numpy().reshape(-1, 1), dtype=torch.float)
        else:
            data[self.target_col].scaler = torch.tensor(df_snap[self.scaler_cols].to_numpy().reshape(-1, 2), dtype=torch.float)

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
                    snap_periods_list = snap_periods_list[int(self.max_target_lags - 1):]
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

            else:
                # all snapshot timestamps
                snap_periods_list = sorted(df[self.time_index_col].unique(), reverse=False)

                # restrict samples for very large datasets based on interleaving
                if df_type == 'train':
                    snap_periods_list = snap_periods_list[int(self.max_target_lags - 1):]
                if (self.interleave > 1) and (df_type == 'train'):
                    snap_periods_list = snap_periods_list[0::self.interleave] + [snap_periods_list[-1]]

                print("picking {} samples for {}".format(len(snap_periods_list), df_type))

                # sample snapshot graphs
                snapshot_list = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(parallel_snapshot_graphs)(df, period) for period in snap_periods_list)

                dataset = DataLoader(snapshot_list, batch_size=self.batch, shuffle=self.shuffle)

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

        infer_df = infer_df.groupby(self.id_col, sort=False).apply(lambda x: x[-1:]).reset_index(drop=True)
        print(infer_df[self.time_index_col].unique().tolist())

        infer_df = infer_df[[self.id_col, 'key_level', self.target_col, self.time_index_col] +
                            self.static_cat_col_list + self.global_context_col_list +
                            self.scaler_cols + self.tweedie_p_col]
        
        model_output = model_output.reshape(-1, 1)
        output = pd.DataFrame(data=model_output, columns=['forecast'])
        
        # merge forecasts with infer df
        output = pd.concat([infer_df, output], axis=1)    

        return output
        
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
              model_dim=128,
              num_layers=1,
              forecast_quantiles=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9],
              dropout=0,
              skip_connection=True,
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
                           dropout=dropout,
                           tweedie_out=self.tweedie_out,
                           skip_connection=skip_connection)

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
              tweedie_loss=False,
              tweedie_variance_power=1.5,
              use_amp=True,
              use_lr_scheduler=True,
              scheduler_params={'factor': 0.5, 'patience': 3, 'threshold': 0.0001, 'min_lr': 0.00001},
              sample_weights=False):

        self.tweedie_loss = tweedie_loss

        if self.tweedie_loss:
            loss_fn = TweedieLoss()
        else:
            loss_fn = QuantileLoss(quantiles=self.forecast_quantiles)
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

                if not self.estimate_tweedie_p:
                    tvp = torch.tensor(tweedie_variance_power)
                    tvp = torch.reshape(tvp, (-1, 1)).to(self.device)
                else:
                    tvp = batch[self.target_col].tvp
                    tvp = torch.reshape(tvp, (-1, 1))

                # compute loss masking out N/A targets -- last snapshot
                if self.tweedie_loss:
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
                if self.tweedie_loss:
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

                    if not self.estimate_tweedie_p:
                        tvp = torch.tensor(tweedie_variance_power)
                        tvp = torch.reshape(tvp, (-1, 1)).to(self.device)
                    else:
                        tvp = batch[self.target_col].tvp
                        tvp = torch.reshape(tvp, (-1, 1))

                    # compute loss masking out N/A targets -- last snapshot
                    if self.tweedie_loss:
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
                    if self.tweedie_loss:
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

                if not self.estimate_tweedie_p:
                    tvp = torch.tensor(tweedie_variance_power)
                    tvp = torch.reshape(tvp, (-1,1)).to(self.device)
                else:
                    tvp = batch[self.target_col].tvp
                    tvp = torch.reshape(tvp, (-1, 1))

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(batch.x_dict, batch.edge_index_dict)

                    # compute loss masking out N/A targets -- last snapshot
                    if self.tweedie_loss:
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
                    if self.tweedie_loss:
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

                    if not self.estimate_tweedie_p:
                        tvp = torch.tensor(tweedie_variance_power)
                        tvp = torch.reshape(tvp, (-1, 1)).to(self.device)
                    else:
                        tvp = batch[self.target_col].tvp
                        tvp = torch.reshape(tvp, (-1, 1))

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(batch.x_dict, batch.edge_index_dict)

                        # compute loss masking out N/A targets -- last snapshot
                        if self.tweedie_loss:
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
                        if self.tweedie_loss:
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

    def infer(self, infer_start, infer_end, select_quantile):

        base_df = self.onetime_prep_df.copy()

        # get list of infer periods
        infer_periods = sorted(
            base_df[(base_df[self.time_index_col] >= infer_start) & (base_df[self.time_index_col] <= infer_end)][
                self.time_index_col].unique().tolist())

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

        for i, t in enumerate(infer_periods):

            print("forecasting period {} at lag {}".format(t, i))

            # infer dataset creation
            infer_df, infer_dataset = self.create_infer_dataset(base_df, infer_till=t)
            output = infer_fn(self.model, self.best_model, infer_dataset)

            # select output quantile
            output_arr = output[0]
            output_arr = output_arr.cpu().numpy()

            # quantile selection
            min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

            assert select_quantile >= min_qtile and select_quantile <= max_qtile, "selected quantile out of bounds!"

            if self.tweedie_loss:
                output_arr = output_arr[:, :, 0]
                output_arr = np.exp(output_arr)
            else:
                try:
                    q_index = self.forecast_quantiles(select_quantile)
                    output_arr = output_arr[:, :, q_index]
                except:
                    q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                    q_lower = int(q_upper - 1)
                    q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower]) / (
                                self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                    q_lower_weight = 1 - q_upper_weight
                    output_arr = q_upper_weight * output_arr[:, :, q_upper] + q_lower_weight * output_arr[:, :, q_lower]

            # show current o/p
            output = self.process_output(infer_df, output_arr)
            # append forecast
            forecast_df = pd.concat([forecast_df, output], axis=0)
            # update df
            base_df = self.update_dataframe(base_df, output)

        if self.log1p_transform:
            forecast_df['forecast'] = np.expm1(forecast_df['forecast'])
            forecast_df[self.target_col] = np.expm1(forecast_df[self.target_col])

        # re-scale output
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler']
        elif self.scaling_method == 'quantile_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_iqr'] + forecast_df[
                'scaler_median']
        else:
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_std'] + forecast_df['scaler_mu']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_std'] + forecast_df[
                'scaler_mu']

        return forecast_df

    def infer_sim(self, infer_start, infer_end, select_quantile, sim_df):

        # get list of infer periods
        infer_periods = sorted(
            sim_df[(sim_df[self.time_index_col] >= infer_start) & (sim_df[self.time_index_col] <= infer_end)][
                self.time_index_col].unique().tolist())

        # print model used for inference
        print("running simulated inference using best saved model: ", self.best_model)

        forecast_df = pd.DataFrame()

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

        for i, t in enumerate(infer_periods):

            print("forecasting period {} at lag {}".format(t, i))

            # infer dataset creation
            infer_df, infer_dataset = self.create_infer_dataset(sim_df, infer_till=t)
            output = infer_fn(self.model, self.best_model, infer_dataset)

            # select output quantile
            output_arr = output[0]
            output_arr = output_arr.cpu().numpy()

            # quantile selection
            min_qtile, max_qtile = min(self.forecast_quantiles), max(self.forecast_quantiles)

            assert select_quantile >= min_qtile and select_quantile <= max_qtile, "selected quantile out of bounds!"

            if self.tweedie_loss:
                output_arr = output_arr[:, :, 0]
                output_arr = np.exp(output_arr)
            else:
                try:
                    q_index = self.forecast_quantiles(select_quantile)
                    output_arr = output_arr[:, :, q_index]
                except:
                    q_upper = next(x for x, q in enumerate(self.forecast_quantiles) if q > select_quantile)
                    q_lower = int(q_upper - 1)
                    q_upper_weight = (select_quantile - self.forecast_quantiles[q_lower]) / (
                            self.forecast_quantiles[q_upper] - self.forecast_quantiles[q_lower])
                    q_lower_weight = 1 - q_upper_weight
                    output_arr = q_upper_weight * output_arr[:, :, q_upper] + q_lower_weight * output_arr[:, :, q_lower]

            # show current o/p
            output = self.process_output(infer_df, output_arr)
            # append forecast
            forecast_df = pd.concat([forecast_df, output], axis=0)
            # update df
            sim_df = self.update_dataframe(sim_df, output)

        if self.log1p_transform:
            forecast_df['forecast'] = np.expm1(forecast_df['forecast'])
            forecast_df[self.target_col] = np.expm1(forecast_df[self.target_col])

        # re-scale output
        if self.scaling_method == 'mean_scaling' or self.scaling_method == 'no_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler']
        elif self.scaling_method == 'quantile_scaling':
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_iqr'] + forecast_df['scaler_median']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_iqr'] + forecast_df[
                'scaler_median']
        else:
            forecast_df['forecast'] = forecast_df['forecast'] * forecast_df['scaler_std'] + forecast_df['scaler_mu']
            forecast_df[self.target_col] = forecast_df[self.target_col] * forecast_df['scaler_std'] + forecast_df[
                'scaler_mu']

        return forecast_df
