import tensorflow as tf

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
    This pipeline loads the images from disk and it rescales them
    """
    return lambda image_path, label: (
        tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3), [img_height, img_width]),
        label
    )
