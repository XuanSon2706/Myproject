import numpy as np
import cv2 as cv
import os
import pickle
import random

img_size = 244
datadir = ("Training")
categories = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

x = []
y = []
for feature, lable in training_data:
    x.append(feature)
    y.append(lable)

x = np.array(x).reshape(-1,img_size,img_size,1)
y = np.array(y)

pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()