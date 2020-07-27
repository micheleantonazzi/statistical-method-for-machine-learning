import shutil
import os
from pyclbr import Function
from typing import List, Dict

import kaggle
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from termcolor import colored
import numpy
import tensorflow as tf


class DataRetrieval:
    def __init__(self, img_width: int = -1, img_height: int = -1):
        self._dataset_name: str = 'moltean/fruits'
        self._categories: List[str] = ['apple', 'banana', 'plum', 'pepper', 'cherry', 'grape', 'tomato', 'potato', 'pear', 'peach']
        self._data_dir: os.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        self._data: Dict[str, numpy.ndarray] = {'images': numpy.array([]), 'labels': numpy.array([])}
        self._img_width = img_width
        self._img_height = img_height

    def download_data(self):
        """
        Download the images for the given categories and save them to disk
        """

        """Verify if the dataset has been already downloaded"""
        if os.path.exists(self._data_dir) and len(os.listdir(self._data_dir)) == 10:
            print(colored('The dataset has been already downloaded', 'green'))
            return

        print(colored('Starting download the dataset', 'red'))

        kaggle.api.authenticate()

        if os.path.exists(self._data_dir):
            shutil.rmtree(self._data_dir)

        kaggle.api.dataset_download_files(self._dataset_name, path=self._data_dir, quiet=False, unzip=True)

        entire_dataset = os.path.join(self._data_dir, 'fruits-360')

        # Dictionary which contains only the images of the selected categories
        filtered_dataset = {category: [] for category in self._categories}

        for sub_path, sub_dirs, files in os.walk(entire_dataset):
            for category in self._categories:
                if '/' + category + ' ' in sub_path.lower():
                    filtered_dataset[category].extend(map(lambda file: os.path.join(sub_path, file), files))
                    break

        # Create the categories' folders and move the relative files
        for category, files in filtered_dataset.items():
            folder = os.path.join(self._data_dir, category)
            os.mkdir(folder)
            count = 0
            for file in files:
                os.rename(file, os.path.join(folder, category + str(count) + '.jpg'))
                count += 1

        """Remove the zip file and the remaining dataset"""
        zip_file = os.path.join(self._data_dir, 'fruits.zip')
        if os.path.exists(zip_file):
            os.remove(zip_file)

        remaining_dataset = os.path.join(self._data_dir, 'fruits-360')
        if os.path.exists(remaining_dataset):
            shutil.rmtree(remaining_dataset)

        print(colored('Dataset downloaded', 'green'))

    def load_dataset(self) -> Dict[str, numpy.ndarray]:
        """
        This method create two numpy array: the first contains the images' paths and the latter contains the their labels
        :return: Dict[str, numpy.array], key 'images' -> numpy.ndarray of the images' paths
                                        key 'labels' -> numpy.ndarray of labels
        """
        images_path = []
        labels = []
        current_label = 0
        for category in self._categories:
            folder = os.path.join(self._data_dir, category)
            for sub_path, sub_dirs, files in os.walk(folder):
                images_path += [os.path.join(folder, file) for file in files]
                labels += [current_label for i in range(len(files))]
            current_label += 1

        images_path = numpy.array(images_path)
        labels = numpy.array(labels)
        self._data = {'images': images_path, 'labels': labels}
        return self._data

    def create_tensorflow_dataset(self, train_set_indices: numpy.ndarray, test_set_indices: numpy.ndarray, map_function: Function):
        """
        Returns two tensorflow dataset optimized for images, each of them containing the images and the labels specified
        in the input parameters (also the order is respected)
        :param train_set_indices: the numpy array indices of the training set
        :param test_set_indices: the numpy array indices of the test set
        :return: a tuple with the tensorflow training and test datasets, optimized for images
        """
        train_set = tf.data.Dataset.from_generator(
            lambda: zip(self._data['images'][train_set_indices], self._data['labels'][train_set_indices]),
            (tf.string, tf.int8)
        ).map(map_function, num_parallel_calls=AUTOTUNE)

        test_set = tf.data.Dataset.from_generator(
            lambda: zip(self._data['images'][test_set_indices], self._data['labels'][test_set_indices]),
            (tf.string, tf.int8)
        ).map(map_function, num_parallel_calls=AUTOTUNE)

        return train_set, test_set

    def get_images_paths(self) -> numpy.ndarray:
        return self._data['images']

    def get_labels(self) -> numpy.ndarray:
        return self._data['labels']

    def get_categories(self) -> List[str]:
        return self._categories














