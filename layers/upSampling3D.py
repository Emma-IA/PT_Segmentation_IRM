import tensorflow as tf
import numpy as np

def upsampling3d(layer_input):
    drop4_t = tf.transpose(layer_input, perm=[3, 0, 1, 2, 4])
    upsampling_result = tf.map_fn(tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear'), drop4_t, infer_shape=False)
    return tf.transpose(upsampling_result, perm=[1, 2, 3, 0, 4])