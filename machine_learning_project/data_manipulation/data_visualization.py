import numpy

from machine_learning_project.data_manipulation.data_retrieval import DataRetrieval
import matplotlib.pyplot as plt

from machine_learning_project.data_manipulation.preprocessing_pipelines import SIMPLEST_PIPELINE


class DataVisualization:
    def __init__(self, data_retrieval: DataRetrieval):
        self._data_retrieval = data_retrieval

    def print_original_images(self):
        shuffled_indices = numpy.random.permutation(len(self._data_retrieval.get_labels()))
        train_set, _ = self._data_retrieval.create_tensorflow_dataset(shuffled_indices, numpy.array([]), SIMPLEST_PIPELINE())
        iterator = iter(train_set)

        plt.figure(figsize=(10, 7))
        for i in range(10):
            while True:
                image, label = next(iterator)
                if label == i:
                    ax = plt.subplot(2, 5, i + 1)
                    plt.imshow(image.numpy().astype("uint8"))
                    plt.title(self._data_retrieval.get_categories()[i].title())
                    plt.axis("off")
                    break
        plt.show()

    def plot_samples_number(self):
        plt.bar(
            numpy.arange(self._data_retrieval.get_n_categories()),
            [numpy.count_nonzero(self._data_retrieval.get_labels() == i) for i in range(self._data_retrieval.get_n_categories())],
            color='royalblue'
        )
        plt.xticks(numpy.arange(self._data_retrieval.get_n_categories()), self._data_retrieval.get_categories())
        plt.xlabel('N. of samples', fontsize=16)
        plt.ylabel('Categories', fontsize=16)
        plt.show()

