#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import gc
import copy
import sklearn
import simplified.BasicGraph as graphmodel
import simplified.HierarchicalGraph as hierarchical_graphmodel
import simplified.MultistepHierarchicalGraph as multistep_hierarchical_graphmodel
import simplified.SmallGraph as small_graphmodel

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
        self.grad_accum = self.data_config.get('grad_accum', False)
        self.accum_iter = self.data_config.get('accum_iter', 1)
        self.scaling_method = self.data_config.get('scaling_method', 'mean_scaling')
        self.fh = self.data_config.get('fh')
        self.forecast = None
        self.baseline_forecast = None

        # check for non-quantile loss fn.
        tweedie_out = self.train_config.get('tweedie_loss', False)
        poisson_out = self.train_config.get('poisson_loss', False)
        rmse_out = self.train_config.get('rmse_loss', False)

        if self.train_batch_size is None:
            self.train_batch_size = 1

        if self.fh is None:
            self.fh = 1

        self.forecast_quantiles = self.model_config.get('forecast_quantiles', None)
        if tweedie_out or poisson_out or rmse_out:
            print("Training for point predictions")
            self.forecast_quantiles = [0.5]
        if self.forecast_quantiles is None:
            print("Training for default quantiles: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]")
            self.forecast_quantiles = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        # default common configs
        self.common_data_config = {'fh': self.fh,
                                   'batch': self.train_batch_size,
                                   'grad_accum': self.grad_accum,
                                   'accum_iter': self.accum_iter,
                                   'scaling_method': self.scaling_method,
                                   'categorical_onehot_encoding': True,
                                   'directed_graph': True,
                                   'shuffle': True,
                                   'interleave': 1}
            
        self.common_model_config = {'forecast_quantiles': self.forecast_quantiles}

        self.data_config.update({k: v for k, v in self.common_data_config.items() if k not in self.data_config})
        self.model_config.update({k: v for k, v in self.common_model_config.items() if k not in self.model_config})
        print("Using data_config: ", self.data_config)
        print("\n")
        print("Using model_config: ", self.model_config)
        print("\n")

        # define & init common data structures
        self.cat_onehot_cols = []
        self.graphobj = None
        self.infer_quantiles = None

    def build(self, data):
        if self.model_type == 'SimpleGraphSage':
            self.graphobj = graphmodel.graphmodel(**self.data_config)
            self.graphobj.build_dataset(data)
            if self.train_config['tweedie_loss']:
                #  remove all forecast quantiles and replace with 1
                self.model_config['forecast_quantiles'] = [0.5]
                print("modifying model_config for tweedie_loss")
            self.graphobj.build(**self.model_config)
            self.infer_quantiles = self.infer_config['select_quantile']
            if len(self.infer_quantiles) == 0 or self.train_config['tweedie_loss']:
                self.infer_quantiles = [0.5]

        elif self.model_type == 'HierarchicalGraphSage':
            self.graphobj = hierarchical_graphmodel.graphmodel(**self.data_config)
            self.graphobj.build_dataset(data)
            if self.train_config['tweedie_loss']:
                #  remove all forecast quantiles and replace with 1
                self.model_config['forecast_quantiles'] = [0.5]
                print("modifying model_config for tweedie_loss")
            self.graphobj.build(**self.model_config)
            self.infer_quantiles = self.infer_config['select_quantile']
            if len(self.infer_quantiles) == 0 or self.train_config['tweedie_loss']:
                self.infer_quantiles = [0.5]

        elif self.model_type == 'MultistepHierarchicalGraphSage':
            self.graphobj = multistep_hierarchical_graphmodel.graphmodel(**self.data_config)
            self.graphobj.build_dataset(data)
            if self.train_config['tweedie_loss']:
                #  remove all forecast quantiles and replace with 1
                self.model_config['forecast_quantiles'] = [0.5]
                print("modifying model_config for tweedie_loss")
            self.graphobj.build(**self.model_config)
            self.infer_quantiles = self.infer_config['select_quantile']
            if len(self.infer_quantiles) == 0 or self.train_config['tweedie_loss']:
                self.infer_quantiles = [0.5]

        elif self.model_type == 'SmallGraphSage':
            self.graphobj = small_graphmodel.graphmodel(**self.data_config)
            self.graphobj.build_dataset(data)
            if self.train_config['tweedie_loss']:
                #  remove all forecast quantiles and replace with 1
                self.model_config['forecast_quantiles'] = [0.5]
                print("modifying model_config for tweedie_loss")
            self.graphobj.build(**self.model_config)
            self.infer_quantiles = self.infer_config['select_quantile']
            if len(self.infer_quantiles) == 0 or self.train_config['tweedie_loss']:
                self.infer_quantiles = [0.5]

    def train(self):
        self.graphobj.train(**self.train_config)
    
    def infer(self, infer_start=None, infer_end=None):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            print("cleared train & test datasets to save memory")
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

    def infer_multistep(self, infer_start=None):
        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            print("cleared train & test datasets to save memory")
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            self.infer_config.pop('select_quantile')
            self.infer_config.update({'select_quantile': quantile})
            self.infer_config['infer_start'] = infer_start
            f_df, forecast_cols = self.graphobj.infer(**self.infer_config)
            f_df[forecast_cols] = np.clip(f_df[forecast_cols], a_min=0, a_max=None)

            f_df = f_df.rename(columns={col: col + '_' + str(quantile) for col in forecast_cols})
            f_df_list.append(f_df)

        self.forecast = pd.concat(f_df_list, axis=1)
        self.forecast = self.forecast.T.drop_duplicates().T

        return self.forecast

    def infer_baseline(self, remove_effects_col_list, infer_start=None, infer_end=None):
        # get the onetime prepped df
        baseline_data = self.graphobj.onetime_prep_df.copy()

        # columns to set to zero
        baseline_cat_onehot_cols = []
        baseline_num_cols = []

        for col in remove_effects_col_list:
            if col in self.baseline_col_dict['temporal_known_cat_col_list']:
                onehot_col_prefix = str(col) + '_'
                baseline_cat_onehot_cols += [c for c in baseline_data.columns.tolist() if c.startswith(onehot_col_prefix)]
            else:
                baseline_num_cols += [col]

        # check all "remove_effects_col_list" have been assigned 0
        for col in baseline_num_cols+baseline_cat_onehot_cols:
            baseline_data[col] = np.where(baseline_data[self.graphobj.time_index_col] >= infer_start, 0,
                                          baseline_data[col])
            num_unique = baseline_data[baseline_data[self.graphobj.time_index_col] >= infer_start][col].unique()
            print("Unique values in the baseline period >= {} for feature {}: {}".format(infer_start, col, num_unique))

        # copy infer config
        baseline_infer_config = copy.deepcopy(self.infer_config)
        baseline_infer_config.update({'sim_df': baseline_data})

        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            print("train & test datasets already cleared from memory")
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            baseline_infer_config.pop('select_quantile')
            baseline_infer_config.update({'select_quantile': quantile})
            if (infer_start is None) or (infer_end is None):
                f_df = self.graphobj.infer_sim(**baseline_infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)
            else:
                baseline_infer_config['infer_start'] = infer_start
                baseline_infer_config['infer_end'] = infer_end
                f_df = self.graphobj.infer_sim(**baseline_infer_config)
                f_df['forecast'] = np.clip(f_df['forecast'], a_min=0, a_max=None)

            if len(self.infer_quantiles) == 1:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast'})
            else:
                f_df = f_df.rename(columns={'forecast': 'baseline_forecast_' + str(quantile)})
            f_df_list.append(f_df)

        self.baseline_forecast = pd.concat(f_df_list, axis=1)
        self.baseline_forecast = self.baseline_forecast.T.drop_duplicates().T

        return self.baseline_forecast

    def infer_multistep_baseline(self, remove_effects_col_list, infer_start=None):
        # get the onetime prepped df
        baseline_data = self.graphobj.onetime_prep_df.copy()

        # columns to set to zero
        baseline_cat_onehot_cols = []
        baseline_num_cols = []

        for col in remove_effects_col_list:
            if col in self.baseline_col_dict['temporal_known_cat_col_list']:
                onehot_col_prefix = str(col) + '_'
                baseline_cat_onehot_cols += [c for c in baseline_data.columns.tolist() if c.startswith(onehot_col_prefix)]
            else:
                baseline_num_cols += [col]

        # check all "remove_effects_col_list" have been assigned 0
        for col in baseline_num_cols+baseline_cat_onehot_cols:
            baseline_data[col] = np.where(baseline_data[self.graphobj.time_index_col] >= infer_start, 0,
                                          baseline_data[col])
            num_unique = baseline_data[baseline_data[self.graphobj.time_index_col] >= infer_start][col].unique()
            print("Unique values in the baseline period >= {} for feature {}: {}".format(infer_start, col, num_unique))

        # copy infer config
        baseline_infer_config = copy.deepcopy(self.infer_config)
        baseline_infer_config.update({'sim_df': baseline_data})

        try:
            del self.graphobj.train_dataset, self.graphobj.test_dataset
            gc.collect()
        except:
            print("train & test datasets already cleared from memory")
            pass

        f_df_list = []
        for quantile in self.infer_quantiles:
            baseline_infer_config.pop('select_quantile')
            baseline_infer_config.update({'select_quantile': quantile})
            baseline_infer_config['infer_start'] = infer_start
            f_df, forecast_cols = self.graphobj.infer_sim(**baseline_infer_config)
            f_df[forecast_cols] = np.clip(f_df[forecast_cols], a_min=0, a_max=None)

            f_df = f_df.rename(columns={col: 'baseline_' + col + '_' + str(quantile) for col in forecast_cols})
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

        if self.model_type == 'SimpleGraphSageAuto':

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
                self.graphobj.build_infer_dataset(infer_till=period)
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

        else:
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
