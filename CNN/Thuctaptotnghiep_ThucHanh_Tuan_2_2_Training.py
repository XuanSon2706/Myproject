import keras
from keras.layers import Conv2D, Dense, Flatten, Activation
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
import pickle


x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x/255.0
y = np_utils.to_categorical(y,4)

model = Sequential()

model.add(Conv2D(3,(3,3),input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(9,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(256))

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(optimizer = "SGD",loss = "crossentropy", metrics = ['accuracy'])

model.fit(x, y,batch_size = 20,epochs=50, validation_split=0.1)

model.save('train_model.model')