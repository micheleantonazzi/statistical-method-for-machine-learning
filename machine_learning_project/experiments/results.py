import os
import pickle
import pandas
from sanitize_ml_labels import sanitize_ml_labels


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
