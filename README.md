# Statistical Methods for Machine Learning
The code contained in this repository automatically downloads and organize the dataset, then in executes trains and evaluates the models and plots the relative graphs. The [dataset](https://www.kaggle.com/moltean/fruits) is hosted by [kaggle.com](https://www.kaggle.com/) and it is not included in this repo. Before execute the code, please configure the kaggle's API by the following steps:

* make sure to have Python 3 installed and correctly configured
* install the kaggle API typing in console ```pip3 install kaggle```
* register to the kaggle website
* go to your personal account page and click the button "Create New API Token"
* the file downloaded in the previous step contains the credentials to login to the kaggle website: do not share with anyone
* copy this file in the path ```/home/{username}/.kaggle```

Now, follow the next steps to run the program:

* install pip3 19.0 or later
* configure your system to use the GPU (this improves the performance of tensorflow)
* open a terminal in the project directory
* install dependencies typing ```pip3 install -r requirements.txt```
* finally, for starting the program, digit ```python3 -m machine_learning_project.run_pipeline```

The models results have already been calculated: they are saved in this repository, and the code simply loads them and plots the graphs.
If you want to recalculate the results, delete the folders ```machine_learning_project/experiments/results``` and ```machine_learning_project/experiments/plots``` and run the project. If the graphs are wrong, please rerun the code and check if they are fixed.

A detailed project report is contained in ```latex``` folder.
