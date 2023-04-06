import os
import matplotlib.pyplot as plt
import cv2 as cv

path = 'C:/Users/sonbu/Desktop/LVTN/Dataset_3D/LGG'
dir = 'C:/Users/sonbu/Desktop/LVTN/LGG'

for f in (os.listdir(path)):
    path1 = os.path.join(path,f)
    print(path1)
    name_folder_img = os.path.basename(path1)
    dir1 = os.path.join(dir,name_folder_img)
    if not os.path.exists(dir1):
            os.mkdir(dir1)
    dir2 = dir1 + '/' + 'Images'
    dir3 = dir1 + '/' + 'Masks'
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    if not os.path.exists(dir3):
        os.mkdir(dir3)
    for file in (os.listdir(path1)):
        path2 = os.path.join(path1,file)
        if path2.find('mask') == -1:
            img = cv.imread(path2)
            name1 = os.path.basename(path2)
            cv.imwrite(dir2 + '/' + name1,img)

            name2 = os.path.splitext(os.path.basename(path2))[0]+'_mask.tif'
            path3 = path2[:-4] +'_mask.tif'
            print(path3)
            msk = cv.imread(path3)
            cv.imwrite(dir3 + '/' + name2, msk)
            
            