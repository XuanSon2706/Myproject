import numpy as np
import cv2 as cv
import os

dir = './kaggle_3m'
dir1 = './TCGA/Images/'
dir2 = './TCGA/Masks/'

for folder in (os.listdir(dir)):
    path1 = os.path.join(dir,folder)
    for file in (os.listdir(path1)):
        path2 = os.path.join(path1,file)
        if path2.find('mask') == -1:
            img = cv.imread(path2)
            name1 = os.path.basename(path2)
            print(dir1 + name1)
            cv.imwrite(dir1 + name1, img)
            
            
            
            name2 = os.path.splitext(os.path.basename(path2))[0]+'_mask.tif'
            path3 = path2[:-4] +'_mask.tif'
            msk = cv.imread(path3)
            cv.imwrite(dir2 + name2, msk)