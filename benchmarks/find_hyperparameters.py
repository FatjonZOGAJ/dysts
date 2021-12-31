#!/usr/bin/python
import argparse
import collections
import os
import sys
import traceback

module_paths = [
    os.path.abspath(os.getcwd()),
    os.path.abspath(os.getcwd() + '//rc_chaos//Methods'),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)


from benchmarks.hyperparameter_config import hyperparameter_configs, get_single_config
from dysts.datasets import *

import pandas as pd

from darts import TimeSeries
import darts.models



from models import models
from models.utils import str2bool

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()

# to change
pts_per_period = 100         # 100
hyp_file_ending = ''  # '_DEBUG_DEBUG_DEBUG'
results_path_ending = ''    # can also be '_DEBUG'
# if we have 100 pts_per_period we should tune on validation data, for 15 it may not overfit too much
EVALUATE_VALID = False       # evaluate on separate part of data
N_JOBS = 1 # -1

parser = argparse.ArgumentParser()
parser.add_argument("--hyperparam_config", help="See hyperparameter_config.py", type=str)
parser.add_argument("--test_single_config", help="", type=str2bool, default=False)
args = parser.parse_args()


# TODO: use full data for better fitting? (may overfit hyperparams)
input_path = os.path.dirname(cwd) + f"/dysts/data/train_univariate__pts_per_period_{pts_per_period}__periods_12.json"
network_inputs = [5, 10, int(0.5 * pts_per_period), pts_per_period]  # can't have kernel less than 5

# input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
# pts_per_period = 100
# network_inputs = [5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period]

SKIP_EXISTING = True
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE,
                 darts.utils.utils.SeasonalityMode.MULTIPLICATIVE]
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE
                 ]
time_delays = [3, 5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period,
               int(1.5 * pts_per_period)]
time_delays = [3, 5, 10, int(0.25 * pts_per_period)]
network_outputs = [1, 4]
network_outputs = [1]

import torch

has_gpu = torch.cuda.is_available()
if not has_gpu:
    warnings.warn("No GPU found.")
else:
    warnings.warn("GPU working.")


dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/hyperparameters/hyperparameters_" + dataname + f"{hyp_file_ending}.json"

equation_data = load_file(input_path)

try:
    with open(output_path, "r") as file:
        all_hyperparameters = json.load(file)
except FileNotFoundError:
    all_hyperparameters = dict()

parameter_candidates = dict()

parameter_candidates["ARIMA"] = {"p": time_delays}
parameter_candidates["LinearRegressionModel"] = {"lags": time_delays}
parameter_candidates["RandomForest"] = {"lags": time_delays}#, "lags_exog": [None]}
parameter_candidates["NBEATSModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TCNModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TransformerModel"] = {"input_chunk_length": network_inputs,
                                            "output_chunk_length": network_outputs}
parameter_candidates["RNNModel"] = {
    "input_chunk_length": network_inputs,
    "output_chunk_length": network_outputs,
    "model": ["LSTM"],
    "n_rnn_layers": [2],
    "n_epochs": [200]
}
parameter_candidates["ExponentialSmoothing"] = {"seasonal": season_values}
parameter_candidates["FourTheta"] = {"season_mode": season_values}
parameter_candidates["Theta"] = {"season_mode": season_values}
for model_name in ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "Prophet"]:
    parameter_candidates[model_name] = {}


'''
parameter_candidates['esn'] = {
    "reservoir_size": [1000],
    "sparsity": [0.01],
    "radius": [0.6],
    "sigma_input": [1],
    "dynamics_fit_ratio": [2 / 7],
    "regularization": [0.0],
    "scaler_tt": ['Standard'],
    "solver": ['pinv'],  # "pinv", "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"
    "seed": [1], # list(range(20))
    "resample": [True],
    'W_scaling':[1, 5]
}
'''

if args.hyperparam_config:
    parameter_candidates = dict()
    parameter_candidates[args.hyperparam_config] = hyperparameter_configs[args.hyperparam_config]
    if args.test_single_config:
        parameter_candidates[args.hyperparam_config] = get_single_config(parameter_candidates[args.hyperparam_config])

    results_path_ending += "_" + args.hyperparam_config

failed_combinations = collections.defaultdict(list)
for e_i, equation_name in enumerate(equation_data.dataset):

    print(f'Equation {e_i} {equation_name}', flush=True)

    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_hyperparameters.keys():
        all_hyperparameters[equation_name] = dict()

    if EVALUATE_VALID:
        split_point = int(5 / 6 * len(train_data))
        train_data = train_data[:split_point]

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    for m_i, model_name in enumerate(parameter_candidates.keys()):
        try:
            print(f'Equation {e_i} {equation_name} Model {m_i} {model_name}', flush=True)
            if SKIP_EXISTING and model_name in all_hyperparameters[equation_name].keys():
                print(f"Entry for {equation_name} - {model_name} found, skipping it.")
                continue

            if (args.hyperparam_config and model_name.split('_')[0] in models):
                model = models[model_name.split('_')[0]]()
            elif model_name in models:
                model = models[model_name]()
            else:
                model = getattr(darts.models, model_name)

            # TODO: check if time_index is correct
            if model_name == 'Prophet':
                df = pd.DataFrame(np.squeeze(y_train_ts.values()))
                df.index = pd.DatetimeIndex(y_train_ts.time_index)
                y_train_ts = TimeSeries.from_dataframe(df)
                df = pd.DataFrame(np.squeeze(y_test_ts.values()))
                df.index = pd.DatetimeIndex(y_test_ts.time_index)
                y_test_ts = TimeSeries.from_dataframe(df)

            model_best = model.gridsearch(parameter_candidates[model_name], y_train_ts, val_series=y_test_ts,
                                          metric=darts.metrics.smape,
                                          n_jobs=N_JOBS)    # ATTENTION: under windows n_jobs has to be 1 or it has to be run from commandline

            best_hyperparameters = model_best[1].copy()

            # Write season object to string
            for hyperparameter_name in best_hyperparameters:
                if "season" in hyperparameter_name:
                    best_hyperparameters[hyperparameter_name] = best_hyperparameters[hyperparameter_name].name

            all_hyperparameters[equation_name][model_name] = best_hyperparameters

            # TODO: evaluate like in compute_benchmarks for faster results
        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            traceback.print_exc()
            continue

        # Overwrite to save even if search stops inbetween
        with open(output_path[:-5] + f'{results_path_ending}.json', 'w') as f:
            json.dump(all_hyperparameters, f, indent=4)

with open(output_path[:-5] + f'{results_path_ending}.json', 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)
print('Failed combinations', failed_combinations)
