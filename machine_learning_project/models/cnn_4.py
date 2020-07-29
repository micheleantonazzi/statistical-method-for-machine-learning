import tensorflow as tf
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


def cnn_4():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(8, 2, activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(4, 2, activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(2, 2, activation='relu'),
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


def cnn_4_parameters():
    return {'epochs': 50, 'batch_size': 64}
