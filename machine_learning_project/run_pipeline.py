import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from machine_learning_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval()
data_retrieval.download_data()
data_retrieval.load_images()
