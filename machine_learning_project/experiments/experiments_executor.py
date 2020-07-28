from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tqdm import tqdm

from machine_learning_project.data_manipulation.data_retrieval import DataRetrieval
from machine_learning_project.models.models_functions import MODELS_FUNCTIONS


class ExperimentsExecutor:
    def __init__(self, data: DataRetrieval, holdouts: int = 3, test_set_size: float = 0.2, pipeline=None):
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
        images, labels = self._data.load_dataset()
        splits = StratifiedShuffleSplit(n_splits=self._holdouts, test_size=self._test_set_size, random_state=42)

        for i, (train_index, test_index) in tqdm(enumerate(splits.split(images, labels)), total=self._holdouts, desc="Computing holdouts", dynamic_ncols=True):
            train_set, test_set = self._data.create_tensorflow_dataset(train_index, test_index, self._preprocessing_pipeline)

            # Optimize train a test sets for improve performances
            train_set = train_set.cache().batch(batch_size=256).prefetch(buffer_size=AUTOTUNE)
            test_set = test_set.cache().batch(batch_size=256).prefetch(buffer_size=AUTOTUNE)

            for model_name, model_function in MODELS_FUNCTIONS.items():
                model = model_function()
                model.fit(train_set, validation_data=test_set, epochs=30, batch_size=256)

