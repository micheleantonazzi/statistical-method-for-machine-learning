import tensorflow as tf
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Flatten, Dense, Input


def ffnn():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def ffnn_parameters():
    return {'epochs': 20, 'batch_size': 256}
