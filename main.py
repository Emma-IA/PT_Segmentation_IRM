from train import train
from generator import create_generators2
import cv2
import numpy as np
from model import micro_unet
from tensorflow import keras
from metrics import IoU_metric
import tensorflow as tf
from predict import predict, show_label

def main():

    prediction = predict('./../data/training/patient001/patient001_frame01.nii.gz')
    label = show_label('./../data/training/patient001/patient001_frame01_gt.nii.gz')

    cv2.imwrite('./label.jpg', label[:,:,3])
    cv2.imwrite('./predict.jpg', prediction[:,:,3])

    # model=train()
    # gt, gv = create_generators2()

    # image, label = gt.__getitem__(0)
    # predic = model.predict(image)

    # predic = np.argmax(predic, -1)
    # label = np.argmax(label, -1)
        

    # cv2.imwrite('./label.jpg', label[0,:,:,3]*50)
    # cv2.imwrite('./predict.jpg', predic[0,:,:,3]*50)

if __name__ == "__main__":
    main()