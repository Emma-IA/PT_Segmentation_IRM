from re import L
import keras.backend as K
import tensorflow as tf
from params import *

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