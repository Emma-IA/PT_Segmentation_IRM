from re import L
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Dropout, concatenate, UpSampling3D, Softmax
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import numpy as np
import nibabel as nib

TRAINING_IMAGE_SIZE = VALIDATION_IMAGE_SIZE = (128,128,6)
LEARNING_RATE = 0.05
PRETRAINED_WEIGHTS = None
BATCH_SIZE = 10
NUMBER_OF_CHANNELS = 1
NBR_CLASSES = 4
DIVISIBILITY_FACTOR = 16

class CustomModel(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # print(tf.reduce_max(y_pred))
            # print('---------------------------------------')
            # print(tf.reduce_min(y_pred))
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def IoULoss(targets, inputs, smooth=1e-6):

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
    IoU = 0
    for i in range(NBR_CLASSES):
        IoU = IoU + IoULoss(targets[:,:,:,:,i],inputs[:,:,:,:,i])
    
    result=IoU/NBR_CLASSES
    
    return result

def IoU_metric(targets, inputs, smooth=1e-6):
    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    # print(result.numpy())
    return IoU

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

def predict(image_nii):
    model = micro_unet(input_size=(None, None, None, 1))
    model.load_weights('./my_checkpoint.ckpt')


    image = load_nii(image_nii)[0]
    size=image.shape
    xcrop = size[0]%DIVISIBILITY_FACTOR
    ycrop = size[1]%DIVISIBILITY_FACTOR
    image = image[:size[0]-xcrop,:size[1]-ycrop,:]
    image = np.expand_dims(np.expand_dims(image, 3), 0)

    result = model.predict(image)
    result = (np.argmax(result, -1)[0,:,:,:])*50
    return result

def show_label(label_nii):
    label = load_nii(label_nii)[0]
    size=label.shape
    xcrop = size[0]%DIVISIBILITY_FACTOR
    ycrop = size[1]%DIVISIBILITY_FACTOR
    label = (label[:size[0]-xcrop,:size[1]-ycrop,:])*50
    return label
