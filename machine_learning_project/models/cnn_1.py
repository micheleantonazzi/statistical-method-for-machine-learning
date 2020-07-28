import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Accuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.layers import Rescaling


def cnn_1():
    model = tf.keras.Sequential([
        Rescaling(1./255),
        Conv2D(6, 6, activation='relu'),
        MaxPooling2D(),
        Conv2D(5, 5, activation='relu'),
        MaxPooling2D(),
        Conv2D(4, 4, activation='relu'),
        MaxPooling2D(),
        Conv2D(3, 3, activation='relu'),
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
        metrics=[Accuracy()]
    )

    return model
