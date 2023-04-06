import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = 'C:/Users/sonbu/Desktop/LVTN/Test_dataset/LIDC'
dir = 'C:/Users/sonbu/Desktop/LVTN/Test_dataset/LIDC2/Images'

for file in (os.listdir(path)):
    path1 = os.path.join(path,file)
    name = os.path.basename(path1)
    for file_1 in (os.listdir(path1)):
        path2 = os.path.join(path1,file_1)
        name1 = os.path.basename(path2)
        for file_2 in (os.listdir(path2)):
            path3 = os.path.join(path2,"images")
            for file_3 in (os.listdir(path3)):
                path4 = os.path.join(path3,file_3)
                
                name2 = os.path.basename(path4)
                path5 = ('{}_{}_{}'.format(name, name1, name2))
        
                img = cv.imread(path4)
                cv.imwrite(dir +"/"+ path5,img)