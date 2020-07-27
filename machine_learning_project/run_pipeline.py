import os

from machine_learning_project.data_manipulation.data_visualization import DataVisualization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from machine_learning_project.data_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval(32, 32)
data_retrieval.download_data()
data_retrieval.load_dataset()


