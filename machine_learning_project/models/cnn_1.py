import tensorflow as tf
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


def cnn_1():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(2, 3, activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def cnn_1_parameters():
    return {'epochs': 20, 'batch_size': 256}
