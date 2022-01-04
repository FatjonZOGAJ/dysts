## Main installation
    git clone https://github.com/FatjonZOGAJ/dysts.git
    cd dysts
    git checkout reservoir_computing
    git submodule init
    git submodule update

    python3 -m venv rc
    source rc/bin/activate
    pip install -r req.txt
    pip install -i https://test.pypi.org/simple/ EchoTorch

To run the experiments and get the rank over all dynamical systems, run: 

    python rc_chaos/Methods/Models/esn/esn_rc_dyst_copy.py 

For plotting in figure_forecasting_benchmarks.ipynb we need to clone the following

    https://github.com/williamgilpin/degas

For AutoARIMA we need to install the following:

    conda install -c conda-forge -c pytorch u8darts-all

Could also install u8darts[prophet] and u8darts[pmdarima] through pip, but that threw errors for me.

Additionally: https://github.com/pytorch/pytorch/issues/35803

    mv C:\Users\blend\miniconda3\envs\rc\lib\site-packages\torch\lib\caffe2_detectron_ops.dll C:\Users\blend\miniconda3\envs\rc\lib\site-packages\torch\lib\caffe2_detectron_ops.dll_old

# Rerunning Experiments

We provide bash scripts to reproduce our experiments and results.
After some manual hyperparameter evaluation we have identified multiple Reservoir Computing and RNN models over which we run an extensive hyperparameter search. 

    bash scripts/test_find_hyperparameter_cells.sh

Note that this will only run a single hyperparameter configuration (indicated by _--test_single_config 1_) as rerunning all experiments takes significant computational resources.
Commands to execute the full experiments have been provided in the latter part of the file (indicated by _bsub_).

Other experiments (including some which were not followed further due to bad performance) can be run by calling:

    bash scripts/default_test.sh

## Dysts Installation

Install from PyPI

    pip install dysts

To obtain the latest version, including new features and bug fixes, download and install the project repository directly from GitHub

    git clone https://github.com/williamgilpin/dysts
    cd dysts
    pip install -I . 

Test that everything is working

    python -m unittest

Alternatively, to use this as a regular package without downloading the full repository, install directly from GitHub

    pip install git+git://github.com/williamgilpin/dysts

The key dependencies are

+ Python 3+
+ numpy
+ scipy
+ pandas
+ sdeint (optional, but required for stochastic dynamics)
+ numba (optional, but speeds up generation of trajectories)

These additional optional dependencies are needed to reproduce some portions of this repository, such as benchmarking experiments and estimation of invariant properties of each dynamical system:

+ nolds (used for calculating the correlation dimension)
+ darts (used for forecasting benchmarks)
+ sktime (used for classification benchmarks)
+ tsfresh (used for statistical quantity extraction)
+ pytorch (used for neural network benchmarks)


## Benchmarks

The benchmarks reported in our preprint can be found in [`benchmarks`](benchmarks/). An overview of the contents of the directory can be found in [`BENCHMARKS.md`](benchmarks/BENCHMARKS.md), while individual task areas are summarized in corresponding Jupyter Notebooks within the top level of the directory.

## Contents

+ Code to generate benchmark forecasting and training experiments are included in [`benchmarks`](benchmarks/)
+ Pre-computed time series with training and test partitions are included in [`data`](dysts/data/)
+ The raw definitions metadata for all chaotic systems are included in the database file [`chaotic_attractors`](dysts/data/chaotic_attractors.json). The Python implementations of differential equations can be found in [`the flows module`](dysts/flows.py)
