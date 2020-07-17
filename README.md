# Statistical Methods for Machine Learning
The [dataset](https://www.kaggle.com/moltean/fruits) is hosted by [kaggle.com](https://www.kaggle.com/) and it is not included in this repo. It is automatically downloaded from the code before. Before doing this, you must configure the kaggle's API by the following steps:

* make sure to have Python 3 installed and correctly configured
* install the kaggle API typing in console ```pip3 install kaggle```
* register to the kaggle website
* go to your personal account page and click the button "Create New API Token"
* the file downloaded in the previous step contains the credentials to login to the kaggle website: do not share with anyone
* copy this file in the path ```/home/{username}/.kaggle```

# Introduction

In the real world there are a lot of different tasks which are too complicated to be modeled by a conventional algorithm. Some problems indeed may have a wide amount of data difficult to analyze. In this case, build a specific algorithm means understand the complex patterns and the hidden correlations between the data. Instead other tasks may be influenced by a lot of external factors which generate a large quantity of similar but different data. These factors are not easy to model, especially considered all together, and often they are not a priori known. This means that an algorithm perform well only in a controlled environment, that respects specific preconditions. On the other hand, if it is applied in the real world, the algorithm may encounter data that it cannot correctly analyze. A particular field of Computer Science is particularly suitable to solve these situations: the machine learning (ML). It represent a family of algorithms that learn automatically through experience. These algorithms are not designed for a specific task but they are general purpose so they can be used to solve each type of task. The principle behind the machine learning is the following: each real phenomenon can be modeled as an unknown mathematical function which can be approximate by a machine learning algorithm. In particular, they build a mathematical model based on sample data, known as training data, to make decisions or predictions without being explicitly programmed to do so. This means that the data play a central role in the machine learning: they must be able to correctly define the model behind the task. First off all, they must be sufficient in number to generalize the problem, especially if the data have a high dimensionality. Secondly, they must be well formed, in terms of range of values, scale and distribution. Often a preprocessing procedure is necessary to modify the data before being used by a learning machine in order to improve its performance. There are three main approaches of machine learning, depending on the nature and the type of the data available to a learning machine. The first is called *supervised learning* and consists in presenting to the learning model the inputs with the correct outputs. The goal is to learn a general function that maps inputs to outputs. Another machine learning technique is *unsupervised learning* where the input are not associated with label, leaving to the learning machine the task of find the data structure. Discovering the hidden patterns of data can be a goal itself or the purpose can be the generation of new data with similar characteristics. The last category is called *reinforcement learning* and occurs when a computer program interacts with a real environment in which it must perform a certain task. The leaning machine is provided by feedback which are analogous to rewards and it tries to maximize them, as it navigates the specific problem space. Machine learning algorithms are often used in computer vision tasks, like object classification, object detection, motion analysis and many others. In the last few years, the state of the art methods to perform object classification use machine learning, in particular *deep learning.* Deep learning is based on artificial neural networks, inspired by biological neural network that composed the animal brains. Specifically to computer vision, the convolutional neural networks (CNNs) are particularly suitable for analyzing images. In fact, they are able to learn how extract the features using the convolutional layers and, subsequently, use these feature in a space invariant way. use   the convolutional layers and subsequently, using this features ind   the best to perform object classification   In particular, the aim of this work is to analyze the in this work is to perform a multi-classification task over images which draw 10 different types of fruit and vegetables. The This work is structured as follow **DA COMPLETARE**

