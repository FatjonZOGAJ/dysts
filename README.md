## Main installation
    git clone https://github.com/FatjonZOGAJ/dysts.git
    cd dysts
    git checkout reservoir_computing

    virtualenv rc --python=python3.7
    source rc/bin/activate
    pip install -r req.txt
    pip install -i https://test.pypi.org/simple/ EchoTorch

# Code
We have adapted the code from the https://github.com/williamgilpin/dysts repository which provides the initial implementations of the dynamical chaos systems as well as the original benchmark. 
For reproducibility reasons we have tried to keep the existing code and experiments as similar as possible and have e.g. *not* refactored existing redundant code as it could potentially  change results and introduce errors.
This additionally will allow for easier pull requests when we push our changes to the original repository. For that reason we ask for leniency regarding code quality.

# Rerunning Experiments

We provide bash scripts to reproduce our experiments and results.
After some manual hyperparameter evaluation we have identified multiple Reservoir Computing and RNN models over which we run an extensive hyperparameter search.

    bash scripts/test_find_hyperparameter_cells.sh

Note that this will only run a single hyperparameter configuration (indicated by _--test_single_config 1_) as rerunning all experiments takes significant computational resources.
Commands to execute the full experiments on Euler have been provided in the latter part of the file (indicated by _bsub_).

Other experiments (including some which were not followed further due to bad performance) can be ran by calling:

    bash scripts/default_tests.sh

These were included for completeness reasons.

To get some more info about the results, call the following:
    
    cd benchmarks/results
    python3 read_results.py

This will print out average ranks and scores for the different models over all dynamical systems and also provide an overview about which models performed best and worst.