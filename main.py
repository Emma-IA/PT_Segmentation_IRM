from train import train
from generator import create_generators2
import cv2
import numpy as np
from model import micro_unet


def main():
    
    # gt, var1, header = load_nii('./../data/training/patient001/patient001_frame01.nii.gz')
    # pred, var2, var3 = load_nii('./../data/training/patient001/patient001_frame01_gt.nii.gz')

    # print('-----------------------------------------------------')
    # print(var1)
    # print('-----------------------------------------------------')
    # print(header)
    # print('=====================================================')
    # print(var2)
    # print('-----------------------------------------------------')
    # print(var3)
    # print('-----------------------------------------------------')
    # print('=====================================================')
    # print(gt.shape)
    # print('-----------------------------------------------------')
    # print(header.get_zooms())
    # print('-----------------------------------------------------')

    # model = micro_unet(input_size=(None, None, None, 1))
    # gt, gv = create_generators2()
    # result = model.predict(gt.__getitem__(0)[0])
    # print(result[0,0,0,0,:])

    model=train()
    gt, gv = create_generators2()

    image, label = gt.__getitem__(0)
    predic = model.predict(image)

    predic = np.argmax(predic, -1)
    label = np.argmax(label, -1)
        

    cv2.imwrite('./label.jpg', label[0,:,:,3]*50)
    cv2.imwrite('./predict.jpg', predic[0,:,:,3]*50)


if __name__ == "__main__":
    main()