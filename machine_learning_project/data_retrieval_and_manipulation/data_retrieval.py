import shutil
import os

import kaggle

class DataRetrieval:
    def __init__(self):
        self._dataset_name = 'moltean/fruits'
        self._categories = ['apple', 'banana', 'plum', 'pepper', 'cherry', 'grape', 'tomato', 'potato', 'pear', 'peach']

    def download_data(self):
        """
        Download the images for the given categories and save them to disk
        :return:
        """

        kaggle.api.authenticate()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

        if not os.path.exists(path):
            os.mkdir(path)
        """
        if not os.listdir(path):
            kaggle.api.dataset_download_files(self._dataset_name, path=path, quiet=False, unzip=True)
        """
        entire_dataset = os.path.join(path, 'fruits-360')

        # Dictionary which contains only the images of the selected categories
        filtered_dataset = {category: [] for category in self._categories}

        for sub_path, sub_dirs, files in os.walk(entire_dataset):
            for category in self._categories:
                if category in sub_path.lower():
                    filtered_dataset[category].extend(map(lambda file: os.path.join(sub_path, file), files))
                    break

        # Create the categories' folders and move the relative files
        for category, files in filtered_dataset.items():
            folder = os.path.join(path, category)
            if not os.path.exists(folder):
                os.mkdir(folder)
            count = 0
            for file in files:
                os.rename(file, os.path.join(folder, category + str(count)))
                count++

        """Remove the zip file and the remaining dataset"""
        zip_file = os.path.join(path, 'fruits.zip')
        if os.path.exists(zip_file):
            os.remove(zip_file)

        remaining_dataset = os.path.join(path, 'fruits-360')
        if os.path.exists(remaining_dataset):
            shutil.rmtree(remaining_dataset)



