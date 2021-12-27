import tensorflow as tf
import keras.backend as K


def IoU_metric(targets, inputs, smooth=1e-6):
    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU
