import collections
import os
import random
import warnings

import darts
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
import copy

from benchmarks.results.read_results import ResultsObject
from dysts.datasets import load_file

NUM_RANDOM_RESTARTS = 15

def eval_all_dyn_syst(model):
    cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = os.getcwd()
    #input_path = os.path.dirname(cwd) + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
    input_path = os.path.dirname(cwd)  + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    #dataname_test =  os.path.splitext(os.path.basename(os.path.split(input_path_test)[-1]))[0]
    
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train") 
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        # 'mase', # requires scaling with train partition; difficult to report accurately
        'mse',
        # 'ope', # runs into issues with zero handling
        'r2_score',
        'rmse',
        # 'rmsle', # requires positive only time series
        'smape'
    ]
    equation_data = load_file(input_path)
    model_name = model.model_name
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)
    
    # eval problematic:
    problematic_equations = ['Arneodo', 'ArnoldWeb', 'BlinkingRotlet', 'BlinkingVortex', 'Bouali', 'CaTwoPlusQuasiperiodic', 'CellularNeuralNetwork', 'Chen', 'CoevolvingPredatorPrey', 'Coullet', 'DequanLi', 'DoubleGyre', 'Duffing', 'ForcedBrusselator']
    
    #for equation_name in equation_data.dataset:
    for equation_name in problematic_equations:

        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        split_point1 = 800
        split_point2 = 1000
        #split_point= int(5 / 6 * len(train_data))
        y_train, y_val, y_test = train_data[:split_point1], train_data[split_point1:split_point2], train_data[split_point2:]
        y_train_val = train_data[:split_point2]
        
        #y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
        
        y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
        y_val_ts = TimeSeries.from_dataframe(pd.DataFrame(y_val))
        y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(y_test))
        y_train_val_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train_val))
        
        try:
            
            if model.model_name == 'RC-CHAOS-ESN-1':

                min_smape = 1000000
                min_smape_model = None

                for i in range(15):

                    hyperparams = model.sample_set_hyperparams()
                    model.resample = True
                    model.fit(y_train_ts)

                    y_val_pred = model.predict(len(y_val))
                    y_val_pred = np.squeeze(y_val_pred.values())
                    
                    #y_val_pred = np.squeeze(model.predict(len(y_val)+len(y_test)).values())

                    pred_y = TimeSeries.from_dataframe(pd.DataFrame(y_val_pred))
                    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

                    metric_func = getattr(darts.metrics.metrics, 'smape')
                    score = metric_func(true_y, pred_y)
                    if score < min_smape:
                        
                        min_hyperparams = hyperparams
                        min_smape = score
                        min_W_in = copy.deepcopy(model.W_in)
                        min_W_h = copy.deepcopy(model.W_h)
                
                model.resample = False
                #model.dynamic_fit_ratio = 4/7
                
                print(min_hyperparams)
                
                model.set_hyperparams(min_hyperparams)
                model.fix_weights(min_W_in, min_W_h)
                model.fit(y_train_val_ts)
                
                y_test_pred = model.predict(len(y_test))
                y_test_pred = np.squeeze(y_test_pred.values())
               

            else:
                model.fit(y_train_val_ts)
                y_test_pred = model.predict(len(y_test))
                y_test_pred = np.squeeze(y_test_pred.values())
        
        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            continue
        
        pred_y = TimeSeries.from_dataframe(pd.DataFrame(y_test_pred))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_test)[:-1]))
        
        print('-----', equation_name)
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            print(metric_name, score)
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

        # TODO: print ranking relative to others for that dynamical system
    print('Failed combinations', failed_combinations)
    results.get_average_rank(model_name, print_out=True)

