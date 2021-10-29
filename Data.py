import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import smart_resize

from Parameters import *
from Utils import *

def create_generators(data_path=DATASET_PATH):
    'Returns three generators'
    image_paths = []
    label_paths = []
    for i in range(1,DATASET_SIZE + 1):
      if i == 100:
        patient = 'patient' + str(i)
      elif i > 9:
        patient = 'patient0' + str(i)
      else:
        patient = 'patient00' + str(i)

      folder_path = os.path.join(data_path, patient)
      image_paths.append(os.path.join(folder_path, patient+'_frame01.nii.gz'))
      image_paths.append(os.path.join(folder_path, patient+'_frame02.nii.gz'))
      label_paths.append(os.path.join(folder_path, patient+'_frame01_gt.nii.gz'))
      label_paths.append(os.path.join(folder_path, patient+'_frame02_gt.nii.gz'))
          

    x_train_list, y_train_list, x_val_list, y_val_list = data_split(image_paths, label_paths)

    train_data_generator = DataGeneratorClassifier(x_train_list, y_train_list,TRAINING_BATCH_SIZE, TRAINING_IMAGE_SIZE)
    validation_data_generator = DataGeneratorClassifier(x_val_list, y_val_list, VALIDATION_BATCH_SIZE, VALIDATION_IMAGE_SIZE, transform=False)
    return train_data_generator, validation_data_generator


def data_split(x_paths_list, y_paths_list):
    'Splits the paths list into three splits'
    # np.random.seed(0)
    # np.random.shuffle(paths_list)
    return x_paths_list[VALIDATION_DATASET_SIZE:], y_paths_list[VALIDATION_DATASET_SIZE:], x_paths_list[:VALIDATION_DATASET_SIZE], y_paths_list[:VALIDATION_DATASET_SIZE]




class DataGeneratorClassifier(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, list_labels, batch_size, image_size, data_path=DATASET_PATH, n_channels=NUMBER_OF_CHANNELS, shuffle=SHUFFLE_DATA, transform=TRANSFORM):
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
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # x_size = []
        # for path in self.list_labels:
        #   x_size.append(len(load_nii(path)[0]))
        # x_size.sort()
        # print(x_size)

        list_IDs_temp = []
        list_labels_temp = []
        for i in range(index*self.batch_size, (index+1)*self.batch_size):
          list_IDs_temp.append(self.list_IDs[i])
          list_labels_temp.append(self.list_labels[i])


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
        X = np.round(np.empty((self.batch_size, *self.image_size, self.n_channels), dtype=int) * 10**(-50),0)
        Y = np.round(np.empty((self.batch_size, *self.image_size, self.n_channels), dtype=int) * 10**(-50),0)

        def fragmentation2(liste, liste_label, nbr_fragment):
          l = len(liste)
          for i in range(nbr_fragment):
            yield liste[int(i*l/nbr_fragment):int((i+1)*l/nbr_fragment)], liste_label[int(i*l/nbr_fragment):int((i+1)*l/nbr_fragment)]

        def moyenne(liste_liste):
            (x,y,z) = liste_liste[0].shape
            n = len(liste_liste)
            nouvelle_liste = np.empty((x,y)) 
            for i in range(x): 
              for j in range(y):
                for liste in liste_liste:
                  nouvelle_liste[i,j] = nouvelle_liste[i,j]+liste[i,j,0]
            return nouvelle_liste/n

        for i in range(len(list_IDs_temp)):

          Xi = load_nii(list_IDs_temp[i])[0]
          Yi = load_nii(list_labels_temp[i])[0]

          liste_slice = []
          liste_slice_label = []
          for z in range(len(Xi[0][0])):
            image_int = Xi[:,:,z]
            image_int = smart_resize(np.expand_dims(image_int, axis=2),(self.image_size[0],self.image_size[1]))
            liste_slice.append(image_int)

            image_int_label = Yi[:,:,z]
            image_int_label = smart_resize(np.expand_dims(image_int_label, axis=2),(self.image_size[0],self.image_size[1]))
            liste_slice_label.append(image_int_label)

          image_redim=[]
          label_redim=[]
          for fragment in fragmentation2(liste_slice, liste_slice_label,self.image_size[2]): 
            image_redim.append(fragment[0][0])
            label_redim.append(fragment[1][0])
            # nouvelle_slice = moyenne(fragment)
            # image_redim.append(nouvelle_slice)

          image_redim = np.asarray(image_redim)
          image_redim = np.swapaxes(image_redim, 0, 1)
          image_redim = np.swapaxes(image_redim, 1, 2)

          label_redim = np.asarray(label_redim)
          label_redim = np.swapaxes(label_redim, 0, 1)
          label_redim = np.swapaxes(label_redim, 1, 2)
        
          X[i:] = image_redim
          Y[i:] = np.round(label_redim,0)

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