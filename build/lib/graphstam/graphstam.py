#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import gc
import copy
import sklearn

# show available model configurations

model_types = {'SimpleGraphSage': "A Basic GNN Model which uses GraphSAGE layers directly on graph structured data (as torch_geometric.data.HeteroData object).\
This model derives basic rolling statistics to use as features alongwith lags & leads.\
This is the fastest & most resource efficient model for datasets of all sizes & should be the first model to try out when experimentng with a new or unfamiliar dataset.",
               'SimpleGraphSageAuto': "Similar to SimpleGraphSageAuto but utilizes LSTM preprocessing layers to automatically extract features from lagged timeseries.",
               'TransformerGraphSage': "This architecture uses a multiheaded self-attention stack to first perform Temporal Attention.\
                                        The attention outputs then act as inputs for the GraphSAGE layers.",
               'TransformerGraphSageLarge': "Similar to TransformerGraphSage but with a more robust, TFT-like multiheaded self-attention stack as the first layer.",
               'GraphSageLSTM': "This architecture performs Spatial (GraphSAGE) operations on temporally organized graph dataset followed by application of LSTM & Self-Attention layers for Temporal learning." 
               }

sample_columns_dict = {'id_col': "Str. Forecast key column",
                       'target_col': "Str. Target variable column",
                       'time_index_col': "Str. Date/Time/Period column corresponding to the required forecast granularity",
                       'global_context_col_list': "List. List of static columns to use for grouping Keys using High-Level similarities.\
                                                   These nodes can help avoid too many p2p edges between key nodes by connecting them via a hub-spoke connection.",
                       'static_cat_col_list': "List. List of static columns to use for grouping Keys using Low-Level similarities.\
                                               These columns result in p2p edges between key nodes.",
                       'temporal_known_num_col_list': "List. Covariate numeric columns with known future values. These will form nodes in the graph & will have directional edges to key nodes.",
                       'temporal_known_cat_col_list': "List. Covariate categorical columns with known future values. These will form nodes in the graph & will have directional edges to key nodes."
                       }


# show sample parameters & arguments for the chosen model

SimpleGraphSage_config = {"data_config": {'col_dict': "A Dictionary describing columns in the dataset",
                                          'max_lags': "Int. No. of lags to consider as part of node features.",
                                          'max_leads': "Int. No. of lead values to consider as part of covariate node features.",
                                          'rolling_window_size': "Int. Window size over which to compute rolling statics for target_col.",
                                          'train_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as training set.",
                                          'test_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as test set.", 
                                          'PARALLEL_DATA_JOBS': "Int. No. of parallel processes to use for various data processing steps. Default: 4", 
                                          'PARALLEL_DATA_JOBS_BATCHSIZE': "Int. No. of items per parallel data job. Default: 128"},
                         
                          "model_config": {'model_dim': "Int. Node embeddings dimension. Use a power-of-2 value like 32, 64, 128 etc.",
                                           'num_layers': "Int. No. of GNN layers to use. Typical range is [1-4]",
                                           'dropout': "Float. Dropout rate [0-1)",
                                           'gnn_skip_connection': "True/False. Whether to use skip connections between GNN layers. Default: False",
                                           'device': "cuda"},
                          
                          "train_config": {'lr': "learning rate.Default: 0.001", 
                                           'min_epochs': "Minimum no. of epochs to train for. Typical range [10-100]", 
                                           'max_epochs': "Maximum no. of epochs to train for. Typical range [100-1000]", 
                                           'patience': "No. of epochs to keep training for without any drop in loss. Typical range [5-20]", 
                                           'min_delta': "Min. change in loss between epochs to be considered as an improvement. Default: 0", 
                                           'model_prefix': "Path & model name prefix to use for saved models. Eg. """"./models_dir/gnn_model"""" ",
                                           'loss_type': "One of ['Quantile','Huber','RMSE']. Default: 'Quantile' ",
                                           'delta': "Applies to 'Huber' loss type. Default: 1",
                                           'use_lr_scheduler': "True/False. Whether to use lr scheduler to reduce learning rate on plateau. Default: True", 
                                           'scheduler_params': "A dict supplying parameters for the scheduler. Default: {'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001} ",
                                           'sample_weights': "True/False. Whether to perform weighted training (weights derived from target_col). Default: False"},
                          
                          "infer_config": {'infer_start': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) from which to start forecasting.", 
                                           'infer_end': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) at which to stop forecasting.", 
                                           'select_quantile': "List of quantiles to forecast if the loss_type is 'Quantile'."}
                                          
                            }


SimpleGraphSageAuto_config = {"data_config": {'col_dict': "A Dictionary describing columns in the dataset",
                                          'max_lags': "Int. No. of lags to consider as part of node features.",
                                          'max_leads': "Int. No. of lead values to consider as part of covariate node features.",
                                          'train_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as training set.",
                                          'test_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as test set.", 
                                          'PARALLEL_DATA_JOBS': "Int. No. of parallel processes to use for various data processing steps. Default: 4", 
                                          'PARALLEL_DATA_JOBS_BATCHSIZE': "Int. No. of items per parallel data job. Default: 128"},
                         
                          "model_config": {'model_dim': "Int. Node embeddings dimension. Use a power-of-2 value like 32, 64, 128 etc.",
                                           'num_layers': "Int. No. of GNN layers to use. Typical range is [1-4]",
                                           'dropout': "Float. Dropout rate [0-1)",
                                           'gnn_skip_connection': "True/False. Whether to use skip connections between GNN layers. Default: False",
                                           'device': "cuda"},
                          
                          "train_config": {'lr':"learning rate.Default: 0.001", 
                                           'min_epochs': "Minimum no. of epochs to train for. Typical range [10-100]", 
                                           'max_epochs': "Maximum no. of epochs to train for. Typical range [100-1000]", 
                                           'patience': "No. of epochs to keep training for without any drop in loss. Typical range [5-20]", 
                                           'min_delta': "Min. change in loss between epochs to be considered as an improvement. Default: 0", 
                                           'model_prefix': "Path & model name prefix to use for saved models. Eg. """"./models_dir/gnn_model"""" ",
                                           'loss_type': "One of ['Quantile','Huber','RMSE']. Default: 'Quantile' ",
                                           'delta': "Applies to 'Huber' loss type. Default: 1",
                                           'use_lr_scheduler': "True/False. Whether to use lr scheduler to reduce learning rate on plateau. Default: True", 
                                           'scheduler_params':"A dict supplying parameters for the scheduler. Default: {'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001} ",
                                           'sample_weights': "True/False. Whether to perform weighted training (weights derived from target_col). Default: False"},
                          
                          "infer_config": {'infer_start': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) from which to start forecasting.", 
                                           'infer_end': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) at which to stop forecasting.", 
                                           'select_quantile': "List of quantiles to forecast if the loss_type is 'Quantile'."}
                                          
                            }


# show sample parameters & arguments for the chosen model

TransformerGraphSage_config = {"data_config": {'col_dict': "A Dictionary describing columns in the dataset",
                                          'max_lags': "Int. No. of lags to consider as part of node features.",
                                          'train_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as training set.",
                                          'test_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as test set.", 
                                          'PARALLEL_DATA_JOBS': "Int. No. of parallel processes to use for various data processing steps. Default: 4", 
                                          'PARALLEL_DATA_JOBS_BATCHSIZE': "Int. No. of items per parallel data job. Default: 128"},
                         
                          "model_config": {'model_dim': "Int. Node embeddings dimension. Use a power-of-2 value like 32, 64, 128 etc.",
                                           'num_layers': "Int. No. of GNN layers to use. Typical range is [1-4]",
                                           'rnn_layers': "Int. No. of LSTM Layers in the encoder layer. Typical range [1-2]",
                                           'attn_layers': "Int. No. of self-attention layers in the encoder layer. Typical range [1-2]",
                                           'temporal_attention_heads': "Int. No. of heads to use in attention layers. Default: 1. Must be a factor of model_dim.",
                                           'dropout': "Float. Dropout rate [0-1)",
                                           'gnn_skip_connection': "True/False. Whether to use skip connections between GNN layers. Default: False",
                                           'device': "cuda"},
                          
                          "train_config": {'lr':"learning rate.Default: 0.001", 
                                           'min_epochs': "Minimum no. of epochs to train for. Typical range [10-100]", 
                                           'max_epochs': "Maximum no. of epochs to train for. Typical range [100-1000]", 
                                           'patience': "No. of epochs to keep training for without any drop in loss. Typical range [5-20]", 
                                           'min_delta': "Min. change in loss between epochs to be considered as an improvement. Default: 0", 
                                           'model_prefix': "Path & model name prefix to use for saved models. Eg. """"./models_dir/gnn_model"""" ",
                                           'loss_type': "One of ['Quantile','Huber','RMSE']. Default: 'Quantile' ",
                                           'delta': "Applies to 'Huber' loss type. Default: 1",
                                           'use_lr_scheduler': "True/False. Whether to use lr scheduler to reduce learning rate on plateau. Default: True", 
                                           'scheduler_params':"A dict supplying parameters for the scheduler. Default: {'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001} ",
                                           'sample_weights': "True/False. Whether to perform weighted training (weights derived from target_col). Default: False"},
                          
                          "infer_config": {'infer_start': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) from which to start forecasting.", 
                                           'infer_end': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) at which to stop forecasting.", 
                                           'select_quantile': "List of quantiles to forecast if the loss_type is 'Quantile'."}
                                          
                            }


# show sample parameters & arguments for the chosen model

GraphSageLSTM_config = {"data_config": {'col_dict': "A Dictionary describing columns in the dataset",
                                        'max_history': "Int. Determines the no. of snapshot graphs in the sequence & hence the size of the overall graph.",
                                        'max_lags': "Int. No. of lags to consider at each snapshot period.",
                                        'max_leads': "Int. No. of leads to consider for covariates at each snapshot period.",
                                        'create_all_temporal_edges': "True/False. Determines if additional temporal edges are required between nodes in different snapshot graphs. Default: False.",
                                        'rolling_window_size': "Int. Window size over which to compute rolling statistics for node features.",
                                        'train_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as training set.",
                                        'test_till': "[Datetime,Str,Int]. Datetime value of the same datatype as target_col until which to consider data as test set.", 
                                        'PARALLEL_DATA_JOBS': "Int. No. of parallel processes to use for various data processing steps. Default: 4", 
                                        'PARALLEL_DATA_JOBS_BATCHSIZE': "Int. No. of items per parallel data job. Default: 128"},
                         
                          "model_config": {'model_dim': "Int. Node embeddings dimension. Use a power-of-2 value like 32, 64, 128 etc.",
                                           'num_layers': "Int. No. of GNN layers to use. Typical range is [1-2]",
                                           'lstm_layers': "Int. No. of LSTM Layers on top of GNN layers. Typical range [1-2]",
                                           'attention_heads': "Int. No. of heads to use in attention layer over lstm_layers o/p. Default: 1. Must be a factor of model_dim.",
                                           'dropout': "Float. Dropout rate [0-1)",
                                           'gnn_skip_connection': "True/False. Whether to use skip connections between GNN layers. Default: False",
                                           'device': "cuda"},
                          
                          "train_config": {'lr':"learning rate.Default: 0.001", 
                                           'min_epochs': "Minimum no. of epochs to train for. Typical range [10-100]", 
                                           'max_epochs': "Maximum no. of epochs to train for. Typical range [100-1000]", 
                                           'patience': "No. of epochs to keep training for without any drop in loss. Typical range [5-20]", 
                                           'min_delta': "Min. change in loss between epochs to be considered as an improvement. Default: 0", 
                                           'model_prefix': "Path & model name prefix to use for saved models. Eg. """"./models_dir/gnn_model"""" ",
                                           'loss_type': "One of ['Quantile','Huber','RMSE']. Default: 'Quantile' ",
                                           'delta': "Applies to 'Huber' loss type. Default: 1",
                                           'use_lr_scheduler': "True/False. Whether to use lr scheduler to reduce learning rate on plateau. Default: True", 
                                           'scheduler_params':"A dict supplying parameters for the scheduler. Default: {'factor':0.5, 'patience':3, 'threshold':0.0001, 'min_lr':0.00001} "},
                          
                          "infer_config": {'infer_start': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) from which to start forecasting.", 
                                           'infer_end': "[Datetime,Str,Int]. Datetime value (of the same datatype as target_col) at which to stop forecasting.", 
                                           'select_quantile': "List of quantiles to forecast if the loss_type is 'Quantile'."}
                                          
                            }

model_configs = {'SimpleGraphSage': SimpleGraphSage_config,
                 'SimpleGraphSageOneshot': SimpleGraphSage_config,
                 'SimpleGraphSageAuto': SimpleGraphSageAuto_config,
                 'SimpleGraphSageAutoOneshot': SimpleGraphSageAuto_config,
                 'TransformerGraphSage': TransformerGraphSage_config, 
                 'TransformerGraphSageLarge': TransformerGraphSage_config, 
                 'GraphSageLSTM': GraphSageLSTM_config}

def usage():
    print('\033[1m'+'Workflow Steps'+'\033[0m')
    print("==============")
    print("\033[1m"+"Step 1. Create a Data Dictionary describing column types in the dataset "+"\033[0m")
    print("... Run show_columns_dict() to get the template for data dictionary.")
    print("... Populate the template dictionary with column names in the pandas dataframe.")
    print("\n")
    print("\033[1m"+"Step 2. Select a Model Type"+"\033[0m")
    print("... Run show_model_types() to get a list of available model types.")
    print("\n")
    print("\033[1m"+"Step 3. For the selected model type, create a config dictionary"+"\033[0m")
    print("... Run show_config_dict(<model_type>) to get the template for the config dict. ")
    print("\n")
    print("\033[1m"+"Step 4. Create a 'gml' object"+"\033[0m")
    print("... Object instantiation takes the two inputs: model_type & config dictionary. For e.g.")
    print("...    gmlobj = gml(model_type=model_type, config=config_dict) ")
    print("\n")
    print("\033[1m"+"Step 5. Build Graph Dataset & Model"+"\033[0m")
    print("... Run gmlobj.build(data) where 'data' is the pandas dataframe")
    print("... Graph datasets for train & test can be retrieved as: trainset, testset = gmlobj.get_datasets")
    print("\n")
    print("\033[1m"+"Step 6. Train Model"+"\033[0m")
    print("... Run gmlobj.train()")
    print("... Model can be retrieved as: model =  gmlobj.graphobj.model")
    print("\n")
    print("\033[1m"+"Step 7. Generate Forecasts"+"\033[0m")
    print("... Run forecasts = gmlobj.infer() . 'forecasts' is the returned pandas dataframe containing predictions.")
    print("\n")
    print("\033[1m"+"Step 8. Run Attribution Analysis"+"\033[0m")
    print("... a) Generate Explanation Objects for each Key Node. For e.g.: gmlobj.generate_explanations(explain_periods=['2023-01-01'], save_dir='./models/')")
    print("... b) Optionally, generate feature importance plots, e.g. gmlobj.show_feature_importance(node_id=None, period=None, topk=20, save_dir='./models/')")
    print("... c) Generate impact of key nodes on each other, e.g. gmlobj.show_key_nodes_importance(node_id=None, period=None, save_dir=None) ")
    print("... d) Generate impact of covariate nodes on key nodes, e.g. gmlobj.show_covariate_nodes_importance(node_id=None, period=None, save_dir='./models/')")
    print("... Note: Steps a, b,c & d write out artefacts such as charts, pickle files, csv files, to the save_dir location for persistence.")

def show_columns_dict():
    print('\033[1m'+'Columns Dictionary Template'+'\033[0m')
    print("===========================")
    print("{")
    for key, desc in sample_columns_dict.items():
        print('\033[1m'+key+'\033[0m'+": ")
        print("   ",desc)
    print("}")
    

def show_model_types():
    print('\033[1m'+'Available Model Types'+'\033[0m')
    print("=====================")
    for model_type, desc in model_types.items():
        print('\033[1m' + model_type + '\033[0m')
        print("   ", desc)
        print("\n")

def show_config_dict(model_type):
    
    print('\033[1m'+'Config Template for {}'.format(model_type)+'\033[0m')
    print("=====================")
    if model_type in ['SimpleGraphSage', 'SimpleGraphSageOneshot']:
        print("   ", json.dumps(SimpleGraphSage_config, indent=4))
    elif model_type in ['SimpleGraphSageAuto', 'SimpleGraphSageAutoOneshot']:
        print("   ", json.dumps(SimpleGraphSageAuto_config, indent=4))
    elif model_type == 'TransformerGraphSage':
        print("   ", json.dumps(TransformerGraphSage_config, indent=4))
    elif model_type == 'TransformerGraphSageLarge':
        print("   ", json.dumps(TransformerGraphSage_config, indent=4))
    elif model_type == 'GraphSageLSTM':
        print("   ", json.dumps(GraphSageLSTM_config, indent=4))
    else:
        raise ValueError("Model Type invalid")
        

class gml(object):

    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        self.data_config = self.config["data_config"]
        self.model_config = self.config["model_config"]
        self.train_config = self.config["train_config"]
        self.infer_config = self.config["infer_config"]
        self.col_dict = self.data_config["col_dict"]
        self.baseline_col_dict = copy.deepcopy(self.col_dict)
        self.train_infer_device = self.model_config.get('device')
        self.train_batch_size = self.data_config.get('batch')
        self.grad_accum = self.data_config.get('grad_accum', True)
        self.accum_iter = self.data_config.get('accum_iter', 1)
        self.scaling_method = self.data_config.get('scaling_method', 'mean_scaling')
        self.fh = self.data_config.get('fh')
        self.forecast = None
        self.baseline_forecast = None

        if self.train_batch_size is None:
            self.train_batch_size = 1

        if self.fh is None:
            self.fh = 1

        if self.train_config.get('loss_type') in ['Huber', 'RMSE']:
            self.forecast_quantiles = [0.5]  # placeholder to make the code work
        elif self.train_config.get('loss_type') == 'Quantile':
            self.forecast_quantiles = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        global graphmodel

        if model_type in ['SimpleGraphSage', 'SimpleGraphSageOneshot']:
            
            import BasicGraph as graphmodel
            # deault common configs
            self.common_data_config = {'fh': self.fh,
                                       'batch': self.train_batch_size,
                                       'grad_accum': self.grad_accum,
                                       'accum_iter': self.accum_iter,
                                       'scaling_method': self.scaling_method,
                                       'categorical_onehot_encoding': True,
                                       'directed_graph': True,
                                       'include_rolling_features': True,
                                       'shuffle': True,
                                       'interleave': 1}
            
            self.common_model_config = {'model_type': "SAGE", 
                                        'model_option': "BASIC", 
                                        'attention_heads': 1,
                                        'forecast_quantiles': self.forecast_quantiles,
                                        'residual_conn_type': 'concat',
                                        'aggr': 'mean',
                                        'use_linear_pretransform': True,
                                        'apply_norm_layers': True,
                                        'use_dirgnn': True}
            
            self.data_config.update(self.common_data_config)
            self.model_config.update(self.common_model_config)

        elif model_type in ['SimpleGraphSageAuto', 'SimpleGraphSageAutoOneshot']:
            
            import BasicGraph as graphmodel
            # deault common configs
            self.common_data_config = {'fh': self.fh,
                                       'batch': self.train_batch_size,
                                       'grad_accum': self.grad_accum,
                                       'accum_iter': self.accum_iter,
                                       'scaling_method': self.scaling_method,
                                       'categorical_onehot_encoding': True,
                                       'directed_graph': True,
                                       'shuffle': True,
                                       'interleave': 1}
            
            self.common_model_config = {'model_type': "SAGE", 
                                        'model_option': "BASIC", 
                                        'attention_heads': 1,
                                        'forecast_quantiles': self.forecast_quantiles,
                                        'residual_conn_type': 'concat',
                                        'aggr': 'mean',
                                        'use_linear_pretransform': True,
                                        'apply_norm_layers': True,
                                        'use_dirgnn': True}
            
            self.data_config.update(self.common_data_config)
            self.model_config.update(self.common_model_config)
            
        elif model_type in ['TransformerGraphSage']:
            
            import TemporalSpatialGraph as graphmodel
            
            self.common_data_config = {'batch': self.train_batch_size,
                                       'grad_accum': self.grad_accum,
                                       'accum_iter': self.accum_iter,
                                       'scaling_method': self.scaling_method,
                                       'categorical_onehot_encoding': True,
                                       'directed_graph': True,
                                       'shuffle': True,
                                       'interleave': 1}
            
            self.common_model_config = {'model_type': "SAGE", 
                                        'model_option': "TEMPORAL_SPATIAL", 
                                        'spatial_attention_heads': 1,
                                        'forecast_quantiles': self.forecast_quantiles,
                                        'residual_conn_type': 'concat',
                                        'aggr': 'mean',
                                        'apply_norm_layers': True,
                                        'use_dirgnn': True}
            
            self.data_config.update(self.common_data_config)
            self.model_config.update(self.common_model_config)

        elif model_type in ['TransformerGraphSageLarge']:

            import TemporalSpatialGraph as graphmodel

            self.common_data_config = {'batch': self.train_batch_size,
                                       'grad_accum': self.grad_accum,
                                       'accum_iter': self.accum_iter,
                                       'scaling_method': self.scaling_method,
                                       'categorical_onehot_encoding': True,
                                       'directed_graph': True,
                                       'shuffle': True,
                                       'interleave': 1}

            self.common_model_config = {'model_type': "SAGE",
                                        'model_option': "TEMPORAL_SPATIAL",
                                        'spatial_attention_heads': 1,
                                        'forecast_quantiles': self.forecast_quantiles,
                                        'residual_conn_type': 'concat',
                                        'aggr': 'mean',
                                        'apply_norm_layers': True,
                                        'use_dirgnn': True}

            self.data_config.update(self.common_data_config)
            self.model_config.update(self.common_model_config)

        elif model_type in ['GraphSageLSTM']:
            
            import SpatialTemporalGraph as graphmodel
            
            # deault common configs
            self.common_data_config = {'fh': 1,
                                       'batch': 1,
                                       'scaling_method': self.scaling_method,
                                       'categorical_onehot_encoding': True,
                                       'directed_graph': True,
                                       'include_rolling_features': True,
                                       'shuffle': True,
                                       'interleave': 1}
            
            self.common_model_config = {'model_type': "SAGE", 
                                        'model_option': "LSTMATTENTION", 
                                        'forecast_quantiles': self.forecast_quantiles,
                                        'residual_conn_type': 'concat',
                                        'aggr': 'mean',
                                        'use_linear_pretransform': True,
                                        'apply_norm_layers': True,
                                        'use_dirgnn': True}
            
            self.data_config.update(self.common_data_config)
            self.model_config.update(self.common_model_config)

            # show scaling method used
            print("Using {} scaling method".format(self.scaling_method))

        else:
            raise ValueError("Not a supported model type!")

    def build(self, data):

        # handle categorical columns
        if len(self.col_dict['temporal_known_cat_col_list']) > 0:
            data = pd.concat([data[self.col_dict['temporal_known_cat_col_list']],
                              pd.get_dummies(data=data,
                                             columns=self.col_dict['temporal_known_cat_col_list'],
                                             prefix_sep='_')], axis=1, join='inner')
            # set all onehot cols created above to zero
            self.cat_onehot_cols = []
            for f in self.col_dict['temporal_known_cat_col_list']:
                onehot_col_prefix = str(f) + '_'
                self.cat_onehot_cols += [c for c in data.columns.tolist() if c.startswith(onehot_col_prefix)]

            # add onehot columns instead of cat cols
            self.col_dict['temporal_known_cat_col_list'] = []
            self.col_dict['temporal_known_num_col_list'] += self.cat_onehot_cols

            #replace the col dict in data_config
            self.data_config.pop('col_dict')
            self.data_config.update({'col_dict':self.col_dict})

        # init graphmodel object
        if self.model_type in ['SimpleGraphSageAuto','TransformerGraphSageLarge']:
            self.graphobj = graphmodel.graphmodel_large(**self.data_config)
        elif self.model_type in ['SimpleGraphSageOneshot']:
            self.graphobj = graphmodel.graphmodel_multihorizon(**self.data_config)
        elif self.model_type in ['SimpleGraphSageAutoOneshot']:
            self.graphobj = graphmodel.graphmodel_large_multihorizon(**self.data_config)
        else:
            self.graphobj = graphmodel.graphmodel(**self.data_config)

        self.graphobj.build_dataset(data)
        self.graphobj.build(**self.model_config)
        self.infer_config.update({'df': data})
        self.infer_quantiles = self.infer_config['select_quantile']
        if len(self.infer_quantiles) == 0:
            self.infer_quantiles = [0.5]
        
    def train(self):
 
        self.graphobj.train(**self.train_config)
    
    def infer(self, infer_start=None, infer_end=None):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.infer_config.pop('select_quantile')
            self.infer_config.update({'select_quantile': quantile})
            if (infer_start is None) or (infer_end is None):
                f_df = self.graphobj.infer(**self.infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)
            else:
                self.infer_config['infer_start'] = infer_start
                self.infer_config['infer_end'] = infer_end
                f_df = self.graphobj.infer(**self.infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)

            if len(self.infer_quantiles) == 1:
                pass
            else:
                f_df = f_df.rename(columns={'forecast': 'forecast_' + str(quantile)})
            f_df_list.append(f_df)

        self.forecast = pd.concat(f_df_list, axis=1)
        self.forecast = self.forecast.T.drop_duplicates().T

        return self.forecast

    def infer_multihorizon(self, infer_start=None):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.infer_config.pop('select_quantile')
            self.infer_config.update({'select_quantile': quantile})
            if (infer_start is None):
                f_df = self.graphobj.infer(df=self.infer_config['df'], infer_start=self.infer_config['infer_start'], select_quantile=self.infer_config['select_quantile'])
                f_df[[f'forecast_{i}' for i in range(self.fh)]] = np.clip(f_df[[f'forecast_{i}' for i in range(self.fh)]], a_min=0, a_max=None)
            else:
                self.infer_config['infer_start'] = infer_start
                f_df = self.graphobj.infer(df=self.infer_config['df'], infer_start=self.infer_config['infer_start'], select_quantile=self.infer_config['select_quantile'])
                f_df[[f'forecast_{i}' for i in range(self.fh)]] = np.clip(f_df[[f'forecast_{i}' for i in range(self.fh)]], a_min=0, a_max=None)

            if len(self.infer_quantiles) == 1:
                pass
            else:
                for i in range(self.fh):
                    f_df = f_df.rename(columns={f'forecast_{i}': f'forecast_{i}' + str(quantile)})
            f_df_list.append(f_df)

        self.forecast = pd.concat(f_df_list, axis=1)

        print("DEBUG: columns in forecast file before drop_duplicate: ", self.forecast.columns.tolist())

        self.forecast = self.forecast.T.drop_duplicates().T

        return self.forecast

    def infer_backtest(self, infer_start=None, infer_end=None):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.infer_config.pop('select_quantile')
            self.infer_config.update({'select_quantile': quantile})
            if (infer_start is None) or (infer_end is None):
                f_df = self.graphobj.backtest(df=self.infer_config['df'], infer_start=self.infer_config['infer_start'], infer_end=self.infer_config['infer_end'], select_quantile=self.infer_config['select_quantile'])
                f_df[[f'forecast_{i}' for i in range(self.fh)]] = np.clip(f_df[[f'forecast_{i}' for i in range(self.fh)]], a_min=0, a_max=None)
            else:
                self.infer_config['infer_start'] = infer_start
                self.infer_config['infer_end'] = infer_end
                f_df = self.graphobj.backtest(df=self.infer_config['df'], infer_start=self.infer_config['infer_start'], infer_end=self.infer_config['infer_end'], select_quantile=self.infer_config['select_quantile'])
                f_df[[f'forecast_{i}' for i in range(self.fh)]] = np.clip(f_df[[f'forecast_{i}' for i in range(self.fh)]], a_min=0, a_max=None)

            if len(self.infer_quantiles) == 1:
                pass
            else:
                for i in range(self.fh):
                    f_df = f_df.rename(columns={f'forecast_{i}': f'forecast_{i}' + str(quantile)})
            f_df_list.append(f_df)

        self.forecast = pd.concat(f_df_list, axis=1)
        self.forecast = self.forecast.T.drop_duplicates().T

        return self.forecast

    def infer_baseline(self, remove_effects_col_list, infer_start=None, infer_end=None):
        # zero-out covariates
        data = self.infer_config['df']
        baseline_data = data.copy()
        # set all onehot cols created above to zero
        baseline_cat_onehot_cols = []
        baseline_num_cols = []
        for col in remove_effects_col_list:
            if col in self.baseline_col_dict['temporal_known_cat_col_list']:
                onehot_col_prefix = str(col) + '_'
                baseline_cat_onehot_cols += [c for c in baseline_data.columns.tolist() if c.startswith(onehot_col_prefix)]
            else:
                baseline_num_cols += [col]

        baseline_data[baseline_num_cols+baseline_cat_onehot_cols] = 0

        baseline_infer_config = copy.deepcopy(self.infer_config)
        baseline_infer_config.pop('df')
        baseline_infer_config.update({'df': baseline_data})

        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            baseline_infer_config.pop('select_quantile')
            baseline_infer_config.update({'select_quantile': quantile})
            if (infer_start is None) or (infer_end is None):
                f_df = self.graphobj.infer(**baseline_infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)
            else:
                baseline_infer_config['infer_start'] = infer_start
                baseline_infer_config['infer_end'] = infer_end
                f_df = self.graphobj.infer(**baseline_infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)

            if len(self.infer_quantiles) == 1:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast'})
            else:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast_' + str(quantile)})
            f_df_list.append(f_df)

        self.baseline_forecast = pd.concat(f_df_list, axis=1)
        self.baseline_forecast = self.baseline_forecast.T.drop_duplicates().T

        return self.baseline_forecast
        
    def get_datasets(self,):
 
        train_dataset, test_dataset = self.graphobj.train_dataset, self.graphobj.test_dataset
        
        return train_dataset, test_dataset
    
    def reset(self,):
        try:
            del self.graphobj
            gc.collect()
        except:
            pass
        self.__init__(self.model_type, self.config)

    def generate_explanations(self, explain_periods, save_dir):

        assert len(explain_periods) > 0, "explain_periods list should have at least one period"
        save_dir = save_dir.rstrip("/")

        import torch
        from torch_geometric.explain import Explainer, CaptumExplainer, ModelConfig, ThresholdConfig, Explanation
        import pickle

        data = self.infer_config['df']

        if self.model_type == 'SimpleGraphSage':

            try:
                del model_config, explainer
                gc.collect()
            except:
                pass

            # Explainer Config
            model_config = ModelConfig(mode="regression", task_level="node", return_type="raw")

            explainer = Explainer(self.graphobj.model,
                                  algorithm=CaptumExplainer('IntegratedGradients'),
                                  explanation_type='model',
                                  node_mask_type='attributes',
                                  edge_mask_type='object',
                                  model_config=model_config)

            # run explanation for period range
            self.explanations_dict = {}

            for period in explain_periods:
                self.graphobj.build_infer_dataset(data, infer_till=period)
                infer_dataset = self.graphobj.infer_dataset
                infer_batch = next(iter(infer_dataset))
                infer_batch = infer_batch.to(self.graphobj.device)

                if period >= self.infer_config['infer_start']:
                    # use forecasts as target
                    target = torch.tensor(
                        self.forecast[self.forecast[self.col_dict['time_index_col']] == period]['forecast'].to_numpy().astype(np.float64))
                else:
                    target = infer_batch[self.col_dict['target_col']].y

                # get node-index map
                node_index_map = self.graphobj.node_index_map

                for node_name, node_index in node_index_map[self.col_dict['id_col']]['index'].items():
                    # run explanation for each node
                    explanation = explainer(x=infer_batch.x_dict,
                                            edge_index=infer_batch.edge_index_dict,
                                            target=target,
                                            index=torch.tensor([node_index]))

                    # save explanation object
                    keyname = str(node_name) + '_' + str(period)
                    filename = save_dir + '/explanation_' + keyname + '.pkl'

                    with open(filename, 'wb') as f:
                        pickle.dump(explanation, f)

                    # save fileloc for the explanation object
                    self.explanations_dict[keyname] = filename
                    print("{} explanation saved.".format(keyname))

        elif self.model_type == 'SimpleGraphSageAuto':

            try:
                del model_config, explainer
            except:
                pass

            # Explainer Config
            model_config = ModelConfig(mode="regression", task_level="node", return_type="raw")

            explainer = Explainer(self.graphobj.model,
                                  algorithm=CaptumExplainer('IntegratedGradients'),
                                  explanation_type='model',
                                  node_mask_type='attributes',
                                  edge_mask_type='object',
                                  model_config=model_config)

            # run explanation for period range
            self.explanations_dict = {}

            for period in explain_periods:
                self.graphobj.build_infer_dataset(data, infer_till=period)
                infer_dataset = self.graphobj.infer_dataset
                infer_batch = next(iter(infer_dataset))
                infer_batch = infer_batch.to(self.graphobj.device)

                if period >= self.infer_config['infer_start']:
                    # use forecasts as target
                    target = torch.tensor(
                        self.forecast[self.forecast[self.col_dict['time_index_col']] == period]['forecast'].to_numpy().astype(np.float64))
                else:
                    target = infer_batch[self.col_dict['target_col']].y

                # get node-index map
                node_index_map = self.graphobj.node_index_map

                for node_name, node_index in node_index_map[self.col_dict['id_col']]['index'].items():
                    if self.train_infer_device == "cuda":
                        # disable cuda to resolve cudnn issue
                        self.graphobj.disable_cuda_backend()
                    else:
                        pass

                    # run explanation for each node
                    explanation = explainer(x=infer_batch.x_dict,
                                            edge_index=infer_batch.edge_index_dict,
                                            target=target,
                                            index=torch.tensor([node_index]))

                    # save explanation object
                    keyname = str(node_name) + '_' + str(period)
                    filename = save_dir + '/explanation_' + keyname + '.pkl'

                    with open(filename, 'wb') as f:
                        pickle.dump(explanation, f)

                    # save fileloc for the explanation object
                    self.explanations_dict[keyname] = filename
                    print("{} explanation saved.".format(keyname))

        elif self.model_type in ['TransformerGraphSage','TransformerGraphSageLarge']:
            raise NotImplementedError


    def show_feature_importance(self, node_id=None, period=None, topk=20, save_dir=None):
        import torch
        from torch_geometric.explain import Explainer, CaptumExplainer, ModelConfig, ThresholdConfig, Explanation
        import pickle

        self.feature_importance_plots_dict = {}

        if node_id is None:
            # generate all feature importances
            for k, v in self.explanations_dict.items():
                with open(v, 'rb') as f:
                    explanation = pickle.load(f)
                if save_dir is not None:
                    filename = save_dir.rstrip("/") + '/' + str(k) + '_feature_importance.png'
                    explanation.visualize_feature_importance(path=filename, top_k=topk,
                                                             feat_labels=self.graphobj.node_features_label)
                    self.feature_importance_plots_dict[k] = filename
                else:
                    explanation.visualize_feature_importance(top_k=topk, feat_labels=self.graphobj.node_features_label)
        else:
            if period is not None:
                keyname = str(node_id) + '_' + str(period)
                explain_file = self.explanations_dict[keyname]
                with open(explain_file, 'rb') as f:
                    explanation = pickle.load(f)

                if save_dir is not None:
                    filename = save_dir.rstrip("/") + '/' + str(keyname) + '_feature_importance.png'
                    explanation.visualize_feature_importance(path=filename, top_k=topk,
                                                             feat_labels=self.graphobj.node_features_label)
                    self.feature_importance_plots_dict[keyname] = filename
                else:
                    explanation.visualize_feature_importance(top_k=topk, feat_labels=self.graphobj.node_features_label)
            else:
                raise ValueError("Provide valid period.")

    def show_key_nodes_importance(self, node_id=None, period=None, save_dir=None):
        import torch
        from torch_geometric.explain import Explainer, CaptumExplainer, ModelConfig, ThresholdConfig, Explanation
        import pickle

        self.impact_nodes_dict = {}

        if node_id is None:
            for k, v in self.explanations_dict.items():
                with open(v, 'rb') as f:
                    explanation = pickle.load(f)

                node_mask_target = explanation.node_mask_dict[self.col_dict['target_col']].sum(
                    dim=1).cpu().numpy()

                topk = node_mask_target.shape[0]
                top_nodes = np.argpartition(node_mask_target, -topk)[-topk:]
                top_node_weights = node_mask_target[top_nodes]

                topn_dict = dict(zip(top_nodes, top_node_weights))
                node_index_map = self.graphobj.node_index_map

                keys_list = list(node_index_map[self.col_dict['id_col']]['index'].keys())
                values_list = list(node_index_map[self.col_dict['id_col']]['index'].values())

                key_wts_dict = {}
                for n, w in topn_dict.items():
                    n_index = values_list.index(n)
                    key = keys_list[n_index]
                    key_wts_dict[key] = abs(w)

                self.impact_nodes_dict[k] = key_wts_dict

            if save_dir is not None:
                filename = save_dir.rstrip("/") + '/' + str(self.col_dict['target_col']) + '_impact_nodes_dict.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(self.impact_nodes_dict, f)

                # write it to a csv file for viz
                impact_nodes_df = pd.DataFrame(self.impact_nodes_dict)
                impact_nodes_df.fillna(0, inplace=True)
                # L1 normalize weights
                norm_wts = sklearn.preprocessing.normalize(impact_nodes_df, norm='l1', axis=0)
                impact_nodes_df[impact_nodes_df.columns.tolist()] = norm_wts

                csv_file = save_dir.rstrip("/") + '/' + str(self.col_dict['target_col']) + '_impact_attribution.csv'
                impact_nodes_df.to_csv(csv_file, index=True)

                print("Key node mutual impact attributions written to file: {}".format(csv_file))

                return impact_nodes_df
            else:
                print(self.impact_nodes_dict)

        else:
            if period is not None:
                keyname = str(node_id) + '_' + str(period)
                explain_file = self.explanations_dict[keyname]
                with open(explain_file, 'rb') as f:
                    explanation = pickle.load(f)

                node_mask_target = explanation.node_mask_dict[self.col_dict['target_col']].sum(
                    dim=1).cpu().numpy()

                topk = node_mask_target.shape[0]
                top_nodes = np.argpartition(node_mask_target, -topk)[-topk:]
                top_node_weights = node_mask_target[top_nodes]

                topn_dict = dict(zip(top_nodes, top_node_weights))
                node_index_map = self.graphobj.node_index_map

                keys_list = list(node_index_map[self.col_dict['id_col']]['index'].keys())
                values_list = list(node_index_map[self.col_dict['id_col']]['index'].values())

                key_wts_dict = {}
                for n, w in topn_dict.items():
                    n_index = values_list.index(n)
                    key = keys_list[n_index]
                    key_wts_dict[key] = sklearn.preprocessing.normalize(abs(w), norm='l1', axis=0)

                self.impact_nodes_dict[keyname] = key_wts_dict
                print(self.impact_nodes_dict)
            else:
                raise ValueError("Provide valid period.")

    def show_covariate_nodes_importance(self, node_id=None, period=None, save_dir=None):
        import torch
        from torch_geometric.explain import Explainer, CaptumExplainer, ModelConfig, ThresholdConfig, Explanation
        import pickle

        self.covariate_nodes_impact_dict = {}

        if node_id is None:
            for k, v in self.explanations_dict.items():
                with open(v, 'rb') as f:
                    explanation = pickle.load(f)

                covar_wt_dict = {}
                for n, w in explanation.node_mask_dict.items():
                    covar_wt_dict[n] = abs(w.sum().cpu().numpy().item())

                self.covariate_nodes_impact_dict[k] = covar_wt_dict

            if save_dir is not None:
                filename = save_dir.rstrip("/") + '/covariate_impact_nodes_dict.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(self.covariate_nodes_impact_dict, f)

                # write it to a csv file for viz
                covariate_nodes_impact_df = pd.DataFrame(self.covariate_nodes_impact_dict)
                covariate_nodes_impact_df.fillna(0, inplace=True)
                # L1 normalize weights
                norm_wts = sklearn.preprocessing.normalize(covariate_nodes_impact_df, norm='l1', axis=0)
                covariate_nodes_impact_df[covariate_nodes_impact_df.columns.tolist()] = norm_wts

                csv_file = save_dir.rstrip("/") + '/covariate_nodes_impact_attribution.csv'
                covariate_nodes_impact_df.to_csv(csv_file, index=True)

                print("Covariate nodes impact attributions written to file: {}".format(csv_file))

                return covariate_nodes_impact_df
            else:
                print(self.covariate_nodes_impact_dict)

        else:
            if period is not None:
                keyname = str(node_id) + '_' + str(period)
                explain_file = self.explanations_dict[keyname]
                with open(explain_file, 'rb') as f:
                    explanation = pickle.load(f)

                covar_wt_dict = {}
                for n, w in explanation.node_mask_dict.items():
                    covar_wts = abs(w.sum().cpu().numpy().item())
                    covar_wt_dict[n] = sklearn.preprocessing.normalize(covar_wts, norm='l1', axis=0)

                self.covariate_nodes_impact_dict[keyname] = covar_wt_dict
                print(self.covariate_nodes_impact_dict)
            else:
                raise ValueError("Provide valid period.")

    def run_attribution_analysis(self, explain_periods=None, save_dir='./'):

        if explain_periods is None:
            explain_periods = sorted(self.forecast[self.col_dict['time_index_col']].unique().tolist(), reverse=False)
        else:
            assert len(explain_periods)>0, "explain_periods should be a list of at least one period to explain."

        # generate explanations
        self.generate_explanations(explain_periods, save_dir)

        # get node wts
        impact_nodes_df = self.show_key_nodes_importance(node_id=None, period=None, save_dir=save_dir)
        impact_nodes_df = impact_nodes_df.reset_index() #.transpose()
        #print(impact_nodes_df.head())
        impact_nodes_df.rename(columns={'index': 'keyname'}, inplace=True)
        impact_nodes_df = impact_nodes_df.set_index('keyname').T.rename_axis('keyname').rename_axis(copy=None, inplace=False).reset_index()
        #print(impact_nodes_df.head())


        # get covar nodes wts
        covariate_nodes_impact_df = self.show_covariate_nodes_importance(node_id=None, period=None, save_dir=save_dir)
        covariate_nodes_impact_df = covariate_nodes_impact_df.reset_index() #.transpose()
        #print(covariate_nodes_impact_df.head())
        covariate_nodes_impact_df.rename(columns={'index': 'keyname'}, inplace=True)
        covariate_nodes_impact_df = covariate_nodes_impact_df.set_index('keyname').T.rename_axis('keyname').rename_axis(copy=None, inplace=False).reset_index()
        #print(covariate_nodes_impact_df.head())

        # forecasts
        forecast = self.forecast
        forecast['keyname'] = forecast[self.col_dict['id_col']].astype(str) + '_' + forecast[self.col_dict['time_index_col']].astype(str)
        forecast = forecast[['keyname', 'forecast', self.col_dict['time_index_col']]]
        #print(forecast.head())

        # transpose & merge all
        attribution_df = impact_nodes_df.merge(covariate_nodes_impact_df, how='inner', on='keyname')
        #print(attribution_df.head())
        attribution_df = attribution_df.merge(forecast, on='keyname', how='inner')
        #print(attribution_df.head())

        # contributions
        covar_columns = list(set(covariate_nodes_impact_df.columns.tolist()) - set(['keyname']))
        attribution_df[covar_columns] = attribution_df[covar_columns].astype(np.float64)
        attribution_df[covar_columns] = attribution_df[covar_columns].multiply(attribution_df['forecast'].astype(np.float64), axis="index")
        attribution_df[self.col_dict['target_col']] = np.abs(attribution_df[self.col_dict['target_col']].astype(np.float64))
        print(attribution_df.head())

        impact_node_columns = list(set(impact_nodes_df.columns.tolist()) - set(['keyname']))
        attribution_df[impact_node_columns] = attribution_df[impact_node_columns].astype(np.float64)
        attribution_df[impact_node_columns] = attribution_df[impact_node_columns].multiply(attribution_df[self.col_dict['target_col']], axis="index")
        print(attribution_df.head())

        return attribution_df
