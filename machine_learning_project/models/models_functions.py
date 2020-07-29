from machine_learning_project.models.cnn_1 import cnn_1, cnn_1_parameters
from machine_learning_project.models.cnn_2 import cnn_2, cnn_2_parameters
from machine_learning_project.models.cnn_3 import cnn_3_parameters, cnn_3

MODELS_FUNCTIONS = {'CNN_1': cnn_1, 'CNN_2': cnn_2, 'CNN_3': cnn_3}
MODELS_PARAMETERS = {'CNN_1': cnn_1_parameters(), 'CNN_2': cnn_2_parameters(), 'CNN_3': cnn_3_parameters()}
