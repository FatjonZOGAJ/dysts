import numpy as np
import pandas as pd
from darts import TimeSeries
from sklearn.ensemble import VotingRegressor

from models.utils import eval_simple, eval_all_dyn_syst
from rc_chaos.Methods.Models.esn.esn_rc_dyst_copy import esn
from rc_chaos.Methods.RUN import new_args_dict


class ESNEnsemble(VotingRegressor):
    def __init__(self, estimators, model_name='RC-CHAOS-ESN-Ensemble'):
        super().__init__(estimators)
        self.model_name = model_name

    def fit(self, X, y=None, sample_weight=None):
        self.training_series = X
        super().fit(X, [], sample_weight)

    def predict(self, X):
        prediction = super().predict(X)
        df = pd.DataFrame(np.squeeze(prediction))
        n = X   # X is actually the number of steps we predict
        df.index = range(len(self.training_series), len(self.training_series) + n)
        return TimeSeries.from_dataframe(df)


def main():
    model_name = 'RC-CHAOS-ESN-Ensemble_DEBUG_DEFAULT'
    kwargs = new_args_dict()
    models = []
    n_models = 5
    for seed in range(n_models):
        kwargs['seed'] = seed
        kwargs['model_name'] = f'RC-CHAOS-ESN-{seed}'
        kwargs['ensemble_base_model'] = True
        models.append((f'esn{seed}', esn(**kwargs)))

    ensemble = ESNEnsemble(models, model_name)

    eval_simple(ensemble)
    eval_simple(esn(**new_args_dict()))
    eval_all_dyn_syst(ensemble)
    eval_all_dyn_syst(esn(**new_args_dict()))


if __name__ == '__main__':
    main()