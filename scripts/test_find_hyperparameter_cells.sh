#!/bin/bash
python benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --test_single_config 1

python benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --test_single_config 1
python benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --test_single_config 1

exit 0;
# can likely also be run for shorter
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --pts_per_period 15

bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN_torch --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_LSTM --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_GRU --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_RNN --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config esn_ESN --pts_per_period 100


bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --pts_per_period 15
bsub -n 12 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --pts_per_period 15

bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_GRU --pts_per_period 100
bsub -n 48 -W 10:00 -R "rusage[mem=512]" python3 benchmarks/find_hyperparameters.py --hyperparam_config RNNModel_RNN --pts_per_period 100

# python rc_chaos/Methods/Models/esn/esn_rc_dyst.py esn_rc_dyst --mode rc_dyst --display_output 0 --system_name Lorenz3D --write_to_log 1 --N 100000 --N_used 1000 --RDIM 1 --noise_level 10 --scaler Standard --approx_reservoir_size 1000 --degree 10 --radius 0.6 --sigma_input 1 --regularization 0.0 --dynamics_length 200 --iterative_prediction_length 500 --num_test_ICS 2 --solver auto --number_of_epochs 1000000 --learning_rate 0.001 --reference_train_time 10 --buffer_train_time 0.5