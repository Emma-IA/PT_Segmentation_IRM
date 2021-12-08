from re import L
import keras.backend as K
import tensorflow as tf
from params import *


def DiceLoss(targets, inputs, smooth=1e-6):
    
    print(targets)
    print('--------------------------------')
    print(inputs)

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

# def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
    
#     BCE =  binary_crossentropy(targets, inputs)
#     intersection = K.sum(K.dot(targets, inputs))    
#     dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     Dice_BCE = BCE + dice_loss
    
#     return Dice_BCE

def IoULoss(targets, inputs, smooth=1e-6):
    # print(targets.shape)
    # print('------------------------------')
    # print(inputs.shape)
    #flatten label and prediction tensors

    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    result = 1 - IoU
    # print(result.numpy())
    return result

def ponderation_IoULoss(targets, inputs, smooth=1e-6):
    # print(targets.shape)
    # print('------------------------------')
    # print(inputs.shape)
    #flatten label and prediction tensors
    IoU = 0
    for i in range(NBR_CLASSES):
        IoU = IoU + IoULoss(targets[:,:,:,:,i],inputs[:,:,:,:,i])
    
    result=IoU/NBR_CLASSES
    
    return result

def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2]*y_true_shape[3], y_true_shape[4]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2]*y_true_shape[3], y_true_shape[4]])

    # [b, classes]
    # count how many of each class are present in 
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)