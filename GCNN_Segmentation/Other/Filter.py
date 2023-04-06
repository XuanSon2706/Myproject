import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('C:/Users/sonbu/Desktop/New folder/3DIRCADB/train/Images/images/image_70.png')

avg_filter = cv.blur(img,(3,3))

median_filter = cv.medianBlur(img,3)

Gauss_filter = cv.GaussianBlur(img,(3,3),0)

plt.figure()
plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(avg_filter)
plt.subplot(143)
plt.imshow(median_filter)
plt.subplot(144)
plt.imshow(Gauss_filter)
plt.show()
