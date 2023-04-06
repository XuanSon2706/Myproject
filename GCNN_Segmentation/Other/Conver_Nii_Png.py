import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


path  = 'C:/Users/sonbu/Desktop/LVTN/seg'
img_png = 'C:/Users/sonbu/Desktop/LVTN/Test_dataset/LiTS17/Masks'

for file in (os.listdir(path)):
    path1 = os.path.join(path,file)
    img = nib.load(path1).get_fdata()
    
    fname = file.replace('.nii','')

   
    name = os.path.basename(fname)

    #img_folder_path = os.path.join(img_png, fname)

    #if not os.path.exists(img_folder_path):
    #    os.mkdir(img_folder_path)
    
    
    for i in range(img.shape[2]):
        slice = img[:, :, i]
        #path2 = img_folder_path + '/' + name + '_{}.png'.format(i)
        path2 = img_png + '/' + name + '_{}.png'.format(i)
        print(path2)
        
        cv.imwrite(path2, slice)
