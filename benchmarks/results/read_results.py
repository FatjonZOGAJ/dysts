import json
from collections import defaultdict


class ResultsObject():
    def __init__(self, path='results_test_univariate__pts_per_period_15__periods_12_DEBUG.json'):
        f = open(path)
        self.results = json.load(f)
        self.new_model_ranks = defaultdict(list)
        self.scores = defaultdict(list)

    def average_all_methods(self):
        models = set()
        for dyn_syst in self.scores:
            for rank, (model_name, score) in enumerate(self.scores[dyn_syst]):
                self.new_model_ranks[model_name].append((dyn_syst, rank))
                models.add(model_name)

        models_ranked = []
        for model in models:
            models_ranked.append((model, self.get_average_rank(model, print_out=True)))

        models_ranked.sort(key = lambda x: x[1])
        for model, rank in models_ranked:
            print(f'Rank {rank:2.3f} {model}')


    def sort_results(self, metric='smape', print_out=False):
        results = self.results
        for dyn_syst in results:
            scores = sorted([(model, results[dyn_syst][model][metric]) for model in results[dyn_syst] if model != 'values'], key = lambda x: x[1])
            self.scores[dyn_syst] = scores
            best = scores[0]
            worst = scores[len(scores) - 1]
            if print_out: print(f'SMAPE {dyn_syst:25} {best[0]:25} {best[1]:10.3f} - {worst[0]:25} {worst[1]:10.03f}')

    def update_results(self,equation_name, model_name, score):
        dyn_sys = self.scores[equation_name]
        i = 0
        while i < len(dyn_sys) and dyn_sys[i][1] < score:
            i += 1
        dyn_sys.insert(i, (model_name, score))

        print('-----', equation_name)
        if i > 0:
            best = dyn_sys[0]
            print(f'\t  0 {best[0]:25} {best[1]:10.3f}')
        if i > 1:
            print(f'\t {i-1:2} {dyn_sys[i-1][0]:25} {dyn_sys[i-1][1]:10.3f}')

        print(f'\t {i:2} {model_name:25} {score:10.3f} \t <----')

        if i < len(dyn_sys) - 2:
            print(f'\t {i+1:2} {dyn_sys[i+1][0]:25} {dyn_sys[i+1][1]:10.3f}')

        if i < len(dyn_sys) - 1:
            worst = dyn_sys[len(dyn_sys)-1]
            print(f'\t {len(dyn_sys)-1:2} {worst[0]:25} {worst[1]:10.3f}')

        self.new_model_ranks[model_name].append((equation_name, i))
        self.scores[equation_name] = dyn_sys
        return i

    def get_average_rank(self, model_name, print_out=True):
        n = len(self.new_model_ranks[model_name])
        rank_sum = sum([x[1] for x in self.new_model_ranks[model_name]])
        avg_rank = rank_sum / n
        n_models = len(self.results['Aizawa']) - 1 # one column is 'values'
        if print_out:
            print(f'{model_name} average rank {avg_rank} out of {n_models} ')
        return avg_rank


if __name__ == "__main__":
    results = ResultsObject()
    results.sort_results(print_out=True)
    results.average_all_methods()
    print('Finished')
