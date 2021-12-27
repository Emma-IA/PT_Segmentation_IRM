from layers.upSampling3D import upsampling3d
from params import *
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Dropout, concatenate, UpSampling3D, Softmax
from tensorflow.keras import Model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from loss import *
from metrics import *
from custommodel import CustomModel
from params import BATCH_SIZE
from tensorflow_addons.layers import InstanceNormalization

def micro_unet(pretrained_weights = PRETRAINED_WEIGHTS,input_size = (TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[1],TRAINING_IMAGE_SIZE[2],NUMBER_OF_CHANNELS)):
#     inputs = tf.random.normal(input_size)
    tf.random.set_seed(1234)
    inputs = Input(input_size, batch_size = BATCH_SIZE)
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = InstanceNormalization()(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(drop3)

    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = InstanceNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = InstanceNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1))(drop5))
    merge6 = concatenate([drop4,up6], axis = 4)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = InstanceNormalization()(conv6)

    up7 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = InstanceNormalization()(conv7)

    up8 = Conv3D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = Conv3D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = InstanceNormalization()(conv9)

    conv10 = Conv3D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = InstanceNormalization()(conv10)
    conv10 = Conv3D(4, 1, activation = 'softmax')(conv10)

    model = CustomModel(inputs = inputs, outputs = conv10, n_gradients=BATCH_SIZE)

    model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = ponderation_IoULoss, metrics = IoU_metric, run_eagerly=True)
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model