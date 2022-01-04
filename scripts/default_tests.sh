#!/bin/bash

# Default RC_ESN
python rc_chaos/Methods/Models/esn/esn_rc_dyst_copy.py

# Ensemble
python models/ESNEnsemble.py

# Backprop ESN
python rc_chaos/Methods/Models/esn/esn_rc_dyst_torch.py