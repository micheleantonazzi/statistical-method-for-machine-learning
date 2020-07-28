import os
import pickle
import pandas
from barplots import barplots
from sanitize_ml_labels import sanitize_ml_labels
import numpy as np
from tabulate import tabulate


class Results:
    def __init__(self):
        self._results = []

    def extract_holdout_results(self, history, preprocessing_pipeline: str, holdout_number: int, model_name: str):
        model_results = []

        scores = pandas.DataFrame(history).iloc[-1].to_dict()
        model_results.append({
            'model': model_name,
            'run_type': 'train',
            'holdout': holdout_number,
            **{
                sanitize_ml_labels(key): value
                for key, value in scores.items()
                if not key.startswith('val_')
            }
        })
        model_results.append({
            'model': model_name,
            'run_type': 'test',
            'holdout': holdout_number,
            **{
                sanitize_ml_labels(key[4:]): value
                for key, value in scores.items()
                if key.startswith("val_")
            }
        })

        self._results = self._results + model_results
        self.save_results(model_results, preprocessing_pipeline, holdout_number, model_name)

    def save_results(self, results, preprocessing_pipeline: str, holdout_number: int, model_name: str):

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', preprocessing_pipeline,
                            'holdout_' + str(holdout_number))
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, model_name + '.pkl')
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    def load_results(self, preprocessing_pipeline: str, holdout_number: int, model_name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', preprocessing_pipeline,
                            'holdout_' + str(holdout_number), model_name + '.pkl')

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._results = self._results + pickle.load(f)
                return True

        return False

    def plot_results(self, preprocessing_pipeline: str):
        results = pandas.DataFrame(self._results)
        results = results.drop(columns=['holdout'])
        height = len(results['model'].unique())
        barplots(
            results,
            groupby=["model", "run_type"],
            show_legend=False,
            height=height,
            orientation="horizontal",
            path='experiments/plots/plots_' + preprocessing_pipeline + '/{feature}.png'
        )

        open('experiments/plots/plots_' + preprocessing_pipeline + '/metrics_table.txt', 'w').close()
        file = open('experiments/plots/plots_' + preprocessing_pipeline + '/metrics_table.txt', 'w')
        models = results.model.unique()
        run_types = results.run_type.unique()
        for metric in ['Accuracy']:
            temp = {run_type: [] for run_type in run_types}
            for model in models:
                for run_type in run_types:
                    res = results[(results['model'] == model) & (results['run_type'] == run_type)][metric].values
                    temp[run_type].append(f'mean = {round(np.mean(res), 4)}\nSTD = {round(np.std(res), 4)}')

            df = pandas.DataFrame({
                'Models': models,
                'Training': temp['train'],
                'Test': temp['test']
            }).set_index('Models')
            file.writelines(f'Table for {preprocessing_pipeline} experiment, metric {metric}\n')
            file.writelines(tabulate(df, tablefmt="pipe", headers="keys") + '\n\n')
        file.close()
