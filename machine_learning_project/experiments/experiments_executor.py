from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from termcolor import colored
from tqdm import tqdm

from machine_learning_project.data_manipulation.data_retrieval import DataRetrieval
from machine_learning_project.experiments.results import Results
from machine_learning_project.models.models_functions import MODELS_FUNCTIONS


from machine_learning_project.models.models_functions import MODELS_PARAMETERS


class ExperimentsExecutor:
    def __init__(self, data: DataRetrieval, holdouts: int = 10, test_set_size: float = 0.2, pipeline=None):
        self._data = data
        self._holdouts = holdouts
        self._test_set_size = test_set_size
        self._preprocessing_pipeline = pipeline

    def set_preprocessing_pipeline(self, pipeline):
        self._preprocessing_pipeline = pipeline

    def set_holdouts(self, holdouts: int):
        self._holdouts = holdouts

    def set_test_set_size(self, test_set_size: float):
        self._test_set_size = test_set_size

    def execute_experiment(self):
        results = Results()
        images, labels = self._data.load_dataset()
        splits = StratifiedShuffleSplit(n_splits=self._holdouts, test_size=self._test_set_size, random_state=42)

        for i, (train_index, test_index) in tqdm(enumerate(splits.split(images, labels)), total=self._holdouts, desc="Computing holdouts", dynamic_ncols=True):
            train_set, test_set = self._data.create_tensorflow_dataset(train_index, test_index, self._preprocessing_pipeline)

            for model_name, model_function in MODELS_FUNCTIONS.items():
                if results.load_results(self._preprocessing_pipeline.__name__, i, model_name):
                    print(colored(f'Holdout {i}: results for {model_name} have already been calculated', 'green'))
                else:
                    print(colored(f'Holdout {i}: training {model_name}', 'red'))
                    parameters = MODELS_PARAMETERS[model_name]

                    # Optimize train a test sets for improve performances
                    train_set_opt = train_set.cache().batch(batch_size=parameters['batch_size']).prefetch(buffer_size=AUTOTUNE)
                    test_set_opt = test_set.cache().batch(batch_size=parameters['batch_size']).prefetch(buffer_size=AUTOTUNE)

                    model = model_function()
                    history = model.fit(train_set_opt, validation_data=test_set_opt, epochs=parameters['epochs']).history

                    # Extract the resulting metric and save them
                    results.extract_holdout_results(history, self._preprocessing_pipeline.__name__, i, model_name)

                    # If it is the first holdout, plot the history graph
                    if i == 0:
                        results.plot_history(history, self._preprocessing_pipeline.__name__, model_name)
        results.plot_results(self._preprocessing_pipeline.__name__)


