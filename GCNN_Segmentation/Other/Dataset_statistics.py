import numpy as np
import cv2 as cv 
import os

n = 0
m = 0
path = 'C:/Users/sonbu/Desktop/LVTN/Dataset/LiTS17/Masks'
for file in (os.listdir(path)):
    path1 = os.path.join(path,file)
    img = cv.imread(path1,cv.IMREAD_GRAYSCALE)
    _,img = cv.threshold(img,127,255,cv.THRESH_BINARY) 
    if img.sum() == 0:
        n += 1
    else:
        m += 1

print('Case of normal: {}'.format(n))
print('Case of unnormal: {}'.format(m))