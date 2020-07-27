import shutil
import os
from typing import List, Dict

import kaggle
from termcolor import colored
import numpy


class DataRetrieval:
    def __init__(self):
        self._dataset_name: str = 'moltean/fruits'
        self._categories: List[str] = ['apple', 'banana', 'plum', 'pepper', 'cherry', 'grape', 'tomato', 'potato', 'pear', 'peach']
        self._data_dir: os.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        self._data: Dict[str, numpy.array] = {'data': None, 'labels': None}

    def collect_data(self):
        pass

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
                if category in sub_path.lower():
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

    def load_dataset(self) -> Dict[str, numpy.array]:
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
        self._data = {'data': images_path, 'labels': labels}
        return self._data











