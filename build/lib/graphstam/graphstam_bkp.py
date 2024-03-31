#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import gc

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
                 'SimpleGraphSageAuto': SimpleGraphSageAuto_config, 
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
    if model_type == 'SimpleGraphSage':
        print("   ", json.dumps(SimpleGraphSage_config, indent=4))
    elif model_type == 'SimpleGraphSageAuto':
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

        if self.train_config.get('loss_type') in ['Huber', 'RMSE']:
            self.forecast_quantiles = [0.5]  # placeholder to make the code work
        elif self.train_config.get('loss_type') == 'Quantile':
            self.forecast_quantiles = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        if model_type in ['SimpleGraphSage']:
            
            import BasicGraph as graphmodel
            # deault common configs
            self.common_data_config = {'fh': 1,
                                       'batch': 1,
                                       'scaling_method': 'mean_scaling',
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
            
        elif model_type in ['SimpleGraphSageAuto']:
            
            import BasicGraph as graphmodel
            # deault common configs
            self.common_data_config = {'fh': 1,
                                       'batch': 1,
                                       'scaling_method': 'mean_scaling',
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
            
        elif model_type in ['TransformerGraphSage','TransformerGraphSageLarge']:
            
            import TemporalSpatialGraph as graphmodel
            
            self.common_data_config = {'scaling_method': 'mean_scaling',
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
                                       'scaling_method': 'mean_scaling',
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
            
            
        else:
            raise ValueError("Not a supported model type!")
            
        
        # init graphmodel object
        self.graphobj = graphmodel.graphmodel(**self.data_config)
            
    def build(self, data):

        self.graphobj.build_dataset(data)
        self.graphobj.build(**self.model_config)
        self.infer_config.update({'df': data})
        self.infer_quantiles = self.infer_config['select_quantile']
        if len(self.infer_quantiles) == 0:
            self.infer_quantiles = [0.5]
        
    def train(self):
 
        self.graphobj.train(**self.train_config)
    
    def infer(self):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.infer_config.pop('select_quantile')
            self.infer_config.update({'select_quantile': quantile})
            f_df = self.graphobj.infer(**self.infer_config)
            f_df['forecast'] == np.clip(f_df['forecast'], a_min=0, a_max=None)
            if len(self.infer_quantiles) == 1:
                pass
            else:
                f_df = f_df.rename(columns={'forecast': 'forecast_' + str(quantile)})
            f_df_list.append(f_df)

        forecast = pd.concat(f_df_list, axis=1)
        forecast = forecast.T.drop_duplicates().T
        return forecast

    def infer_baseline(self, remove_effects_col_list):
        # zero-out covariates
        data = self.infer_config['df']
        baseline_data = data.copy()
        """
        # handle categorical columns
        if len(self.col_dict['temporal_known_cat_col_list']) > 0:
            baseline_data = pd.concat([baseline_data[self.col_dict['temporal_known_cat_col_list']],
                                       pd.get_dummies(data=baseline_data,
                                                      columns=self.col_dict['temporal_known_cat_col_list'],
                                                      prefix_sep='_')],
                                      axis=1, join='inner')
            # set all onehot cols created above to zero
            onehot_cols = []
            for f in self.col_dict['temporal_known_cat_col_list']:
                onehot_col_prefix = str(f) + '_'
                onehot_cols += [c for c in baseline_data.columns.tolist() if c.startswith(onehot_col_prefix)]

            baseline_data[onehot_cols] = 0
            # add onehot columns to
            
        """
        
        baseline_infer_config = self.infer_config.copy(deep=True)
        baseline_infer_config.pop('df')
        baseline_infer_config.update({'df': baseline_data})

        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.baseline_infer_config.pop('select_quantile')
            self.baseline_infer_config.update({'select_quantile': quantile})
            f_df = self.graphobj.infer(**self.baseline_infer_config)
            if len(self.infer_quantiles) == 1:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast'})
            else:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast_' + str(quantile)})
            f_df_list.append(f_df)

        forecast = pd.concat(f_df_list, axis=1)
        forecast = forecast.T.drop_duplicates().T

        return forecast
        
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
    
    def explain(self,):
        raise NotImplementedError
    



