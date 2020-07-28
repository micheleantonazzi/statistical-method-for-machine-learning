import os

from machine_learning_project.data_manipulation.data_visualization import DataVisualization
from machine_learning_project.data_manipulation.preprocessing_pipelines import SCALE_PIPELINE
from machine_learning_project.experiments.experiments_executor import ExperimentsExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from machine_learning_project.data_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval(32, 32)
data_retrieval.download_data()
experiment_executor = ExperimentsExecutor(data_retrieval)
experiment_executor.set_preprocessing_pipeline(SCALE_PIPELINE(32, 32))
experiment_executor.execute_experiment()


