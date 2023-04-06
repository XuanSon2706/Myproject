import numpy as np
import skimage
from skimage.measure import block_reduce
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('C:/Users/sonbu/Desktop/New folder/3DIRCADB/train/Images/images/image_70.png', cv.IMREAD_GRAYSCALE)
img1 = cv.resize(img,(4,4))

max_pooling = block_reduce(img, block_size = (2, 2), func=np.max)
max_pooling1 = block_reduce(max_pooling, block_size = (2, 2), func=np.max)
max_pooling2 = block_reduce(max_pooling1, block_size = (2, 2), func=np.max)

avg_pooling = block_reduce(img, block_size = (2, 2), func=np.mean)
avg_pooling1 = block_reduce(avg_pooling , block_size = (2, 2), func=np.mean)
avg_pooling2 = block_reduce(avg_pooling1 , block_size = (2, 2), func=np.mean)




plt.figure()

plt.subplot(241)
plt.imshow(img)
plt.title("Ảnh gốc")

plt.subplot(242)
plt.imshow(max_pooling)
plt.title("1 lớp Pooling")

plt.subplot(243)
plt.imshow(max_pooling1)
plt.title("2 lớp Pooling")

plt.subplot(244)
plt.imshow(max_pooling2)
plt.title("3 lớp Pooling")

plt.subplot(245)
plt.imshow(img)
plt.title("Ảnh gốc")

plt.subplot(246)
plt.imshow(avg_pooling)
plt.title("1 lớp Pooling")

plt.subplot(247)
plt.imshow(avg_pooling1)
plt.title("2 lớp Pooling")

plt.subplot(248)
plt.imshow(avg_pooling2)
plt.title("3 lớp Pooling")
plt.show()
