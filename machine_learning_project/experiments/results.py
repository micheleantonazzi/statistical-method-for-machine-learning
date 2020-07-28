import os
import pickle


def save_results(results, preprocessing_pipeline: str, holdout_number: int, model_name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', preprocessing_pipeline,
                        'holdout_' + str(holdout_number))
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, model_name + '.pkl')
    with open(os.path.join(path), 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
