import tensorflow as tf
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.python.keras.layers import Rescaling


def cnn_1():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(3, 2, activation='relu'),
        MaxPooling2D(),
        Conv2D(2, 2, activation='relu'),
        MaxPooling2D(),
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
