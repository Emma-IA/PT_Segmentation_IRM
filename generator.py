from params import *
import tensorflow as tf
import os
import numpy as np
from utils import *
from keras.preprocessing.image import smart_resize
import cv2


def create_generators2(data_path=DATASET_PATH):
    'Returns three generators'
    validation_image_paths = []
    validation_label_paths = []
    # for i in range(1,2):
    for i in range(1,VALIDATION_DATASET_SIZE + 1):
      if i == 100:
        patient = 'patient' + str(i)
      elif i > 9:
        patient = 'patient0' + str(i)
      else:
        patient = 'patient00' + str(i)

      folder_path = os.path.join(data_path, patient)
      validation_image_paths.append(os.path.join(folder_path, patient+'_frame01.nii.gz'))
      validation_image_paths.append(os.path.join(folder_path, patient+'_frame02.nii.gz'))
      validation_label_paths.append(os.path.join(folder_path, patient+'_frame01_gt.nii.gz'))
      validation_label_paths.append(os.path.join(folder_path, patient+'_frame02_gt.nii.gz'))

    train_image_paths = []
    train_label_paths = []
    # for i in range(2, 3):
    for i in range(VALIDATION_DATASET_SIZE + 1, DATASET_SIZE + 1):
      if i == 100:
        patient = 'patient' + str(i)
      elif i > 9:
        patient = 'patient0' + str(i)
      else:
        patient = 'patient00' + str(i)

      folder_path = os.path.join(data_path, patient)
      train_image_paths.append(os.path.join(folder_path, patient+'_frame01.nii.gz'))
      train_image_paths.append(os.path.join(folder_path, patient+'_frame02.nii.gz'))
      train_label_paths.append(os.path.join(folder_path, patient+'_frame01_gt.nii.gz'))
      train_label_paths.append(os.path.join(folder_path, patient+'_frame02_gt.nii.gz'))
     
    train_data_generator = DataGeneratorClassifier2(train_image_paths, train_label_paths,TRAINING_BATCH_SIZE, TRAINING_IMAGE_SIZE)
    validation_data_generator = DataGeneratorClassifier2(validation_image_paths, validation_label_paths, VALIDATION_BATCH_SIZE, VALIDATION_IMAGE_SIZE, transform=False)
    return train_data_generator, validation_data_generator


class DataGeneratorClassifier2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, list_labels, batch_size=1, image_size=(None,None,None), data_path=DATASET_PATH, n_channels=NUMBER_OF_CHANNELS, shuffle=SHUFFLE_DATA, transform=TRANSFORM):
        'Initialisation'
        self.classes = os.listdir(data_path)
        self.image_size = image_size
        self.batch_size = BATCH_SIZE
        self.list_IDs = list_IDs
        self.list_labels = list_labels
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_path=data_path
        self.transform=transform

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)))

    def __getitem__(self, index):
        'Generate one batch of data'

        list_IDs_temp = []
        list_labels_temp = []
        # for i in range(index*self.batch_size, (index+1)*self.batch_size):
        #   list_IDs_temp.append(self.list_IDs[i])
        #   list_labels_temp.append(self.list_labels[i])
        list_IDs_temp.append(self.list_IDs[index])
        list_labels_temp.append(self.list_labels[index])


        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # list_IDs_temp = self.list_IDs[indexes]
        # list_labels_temp = self.list_labels[indexes]
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *image_size, n_channels)
        
        # for i in range(len(list_IDs_temp)):
        Xi = load_nii(list_IDs_temp[0])[0]
        Yi = load_nii(list_labels_temp[0])[0]
      
        size=Xi.shape
        xcrop = size[0]%DIVISIBILITY_FACTOR
        ycrop = size[1]%DIVISIBILITY_FACTOR

        Xi = Xi[:size[0]-xcrop,:size[1]-ycrop,:]
        Yi = Yi[:size[0]-xcrop,:size[1]-ycrop,:]

        X = np.expand_dims(np.expand_dims(Xi, 3), 0)
        Y = tf.one_hot(np.expand_dims(Yi, 0).astype(np.int32), NBR_CLASSES, axis=-1)

        # if self.transform:
        #     data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #     layers.experimental.preprocessing.RandomRotation(0.8)])
        #     X=data_augmentation(X)

        return X, Y

def show_batch(generator, batch_number=1):
    images, labels = generator.__getitem__(batch_number)
    fig,ax=plt.subplots(nrows=1, ncols=6)
    slices_indices=[1,2,3,4,5,6]
    for i in range(len(images)):
        for j in range(len(slices_indices)):
            images_slice=images[i,:,:,j,0]
            ax[j].imshow(images_slice,cmap='gray')
            ax[j].axis('off')
    plt.show