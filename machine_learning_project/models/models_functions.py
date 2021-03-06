from machine_learning_project.models.cnn_1 import cnn_1, cnn_1_parameters
from machine_learning_project.models.cnn_2 import cnn_2, cnn_2_parameters
from machine_learning_project.models.cnn_3 import cnn_3_parameters, cnn_3
from machine_learning_project.models.cnn_4 import cnn_4, cnn_4_parameters
from machine_learning_project.models.ffnn import ffnn, ffnn_parameters
from machine_learning_project.models.perceptron import perceptron, perceptron_parameters

MODELS_FUNCTIONS = {'Perceptron': perceptron, 'FFNN': ffnn, 'CNN_1': cnn_1, 'CNN_2': cnn_2, 'CNN_3': cnn_3, 'CNN_4': cnn_4}
MODELS_PARAMETERS = {'Perceptron': perceptron_parameters(), 'FFNN': ffnn_parameters(), 'CNN_1': cnn_1_parameters(), 'CNN_2': cnn_2_parameters(), 'CNN_3': cnn_3_parameters(), 'CNN_4': cnn_4_parameters()}
