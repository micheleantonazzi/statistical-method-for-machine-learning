import os
import pickle
import pandas
from barplots import barplots
from sanitize_ml_labels import sanitize_ml_labels
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


class Results:
    def __init__(self):
        self._results = []
        self._histories = {}

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
            path=os.path.dirname(os.path.abspath(__file__)) + '/plots/plots_' + preprocessing_pipeline + '/{feature}.png'
        )

        path = os.path.dirname(os.path.abspath(__file__)) + '/plots/plots_' + preprocessing_pipeline + '/metrics_table.txt'
        open(path, 'w').close()
        file = open(path, 'w')
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

    def plot_history(self, history, preprocessing_pipeline: str, model_name: str):
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(history['accuracy'], label='accuracy')
        ax1.plot(history['val_accuracy'], label='val_accuracy')
        ax1.set(xlabel='Epoch', ylabel='Accuracy')
        ax1.legend(loc='lower right')
        ax1.set_title(f'{model_name} training accuracy')
        ax2.plot(history['loss'], label='loss')
        ax2.plot(history['val_loss'], label='val_loss')
        ax2.set(xlabel='Epoch', ylabel='Loss')
        ax2.legend(loc='lower left')
        ax2.set_title(f'{model_name} training loss')
        fig.tight_layout()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'plots_' + preprocessing_pipeline)
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, model_name.lower() + '_training_accuracy.png'))

    def execute_wilcoxon_test(self, preprocessing_pipeline: str, alpha: int = 0.01):
        results = pandas.DataFrame(self._results)
        results = results[(results['run_type'] == 'test')]
        models = results.model.unique()
        path = os.path.dirname(os.path.abspath(__file__)) + '/plots/plots_' + preprocessing_pipeline + '/wilcoxon.txt'
        open(path, 'w').close()
        file = open(path, 'w')
        for metric in ['Accuracy', 'Loss']:
            for model_a in models:
                for model_b in models:
                    if not model_a == model_b:
                        model_a_values = results[results['model'] == model_a][metric]
                        model_b_values = results[results['model'] == model_b][metric]
                        stats, p_value = wilcoxon(model_a_values, model_b_values)
                        if p_value > alpha:
                            file.write(f"In {preprocessing_pipeline} experiments, for metric {metric}, {model_a} and {model_b} statistically identical, with a p_value of {p_value}\n")
                        elif not metric == 'Loss':
                            if model_a_values.mean() > model_b_values.mean():
                                file.write(f"In {preprocessing_pipeline} experiments, for metric {metric}, {model_a} is BETTER than {model_b}, with a p_value of {p_value}\n")
                            else:
                                file.write(f"In {preprocessing_pipeline} experiments, for metric {metric}, {model_a} is WORST than {model_b}, with a p_value of {p_value}\n")
                        else:
                            if model_a_values.mean() > model_b_values.mean():
                                file.write(f"In {preprocessing_pipeline} experiments, for metric {metric}, {model_a} is WORST than {model_b}, with a p_value of {p_value}\n")
                            else:
                                file.write(f"In {preprocessing_pipeline} experiments, for metric {metric}, {model_a} is BETTER than {model_b}, with a p_value of {p_value}\n")

            file.write('\n')

