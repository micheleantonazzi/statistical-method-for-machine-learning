import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def SIMPLEST_PIPELINE():
    """
    This is the simples pipeline, it simply loads the image from disk
    """
    return lambda image_path, label: (
        tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3),
        label
    )

def SCALE_PIPELINE(img_width, img_height):
    """
    This pipeline loads the images from disk and it rescales them. After that, it standardize the pixel values
    between the interval of [0, 1]
    """
    normalization = Rescaling(1./255)
    def scale_pipeline(image_path, label):
        return (
            normalization(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3), [img_height, img_width])),
            label
        )

    return scale_pipeline
