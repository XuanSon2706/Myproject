import cv2 as cv
import tkinter as tk
import PIL
import keras
from PIL import ImageTk, Image
from tkinter import *
from tkinter import filedialog as fd

window = Tk()
window.title("Welcome to my project")
frame_2 = Frame(window)
frame_2.grid(column= 0, row = 1,sticky = W)
model = keras.models.load_model('train_model.pb')

def open():
    global img_1,img_2
    filetypes = (('JPEG', ('*.jpg','*.jpeg','*.jpe')), ('PNG', '*.png'),('GIF','*.gif'), ('All files', '*.*'))
    filename = fd.askopenfilename(title='Open', initialdir='/', filetypes=filetypes)
    img_1 = PIL.Image.open(filename)
    (h_1, w_1) = img_1.size
    (h_3, w_3) = img_1.size
    if (h_1 > 500) or (w_1 > 500):
        h_3 = int(h_1 /5)
        w_3 = int(w_1 /5)
        dim = (h_3, w_3)
        img_1=img_1.resize(dim)
    else:
        pass
    img_2 = PIL.ImageTk.PhotoImage(img_1)
    frame_1 = Frame(window, width=w_3, height=h_3)
    frame_1.grid(column=0, row=0)
    img_show = tk.Label(frame_1, image=img_2).pack()

    def prepare(filepath):
        img_size = 244
        img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        new_array = cv.resize(img_array, (img_size, img_size))
        new_array = new_array / 255.0
        return new_array.reshape(-1, img_size, img_size, 1)
    prediction = model.predict([prepare('{}'.format(filename))])

    a = prediction[0]
    c = 0
    if a[0] >= 0.9:
        b = 'Cat'
        c = a[0]
    elif a[1] >= 0.9:
        b = 'Dog'
        c = a[1]
    elif a[2] >= 0.9:
        b = 'Panda'
        c = a[2]
    else:
        b = 'Unknown'
        c = a[3]

    confident_interval = tk.Label(frame_2, text="Confidence: {} %".format(c * 100)).grid(column=2, row=3, pady=8,padx=100,sticky=W)
    categories_lable = tk.Label(frame_2, text="Category: {}".format(b)).grid(column=2, row=2, pady=8,padx=100, sticky=W)
    classification = tk.Label(frame_2, text="After Classification", font='bold').grid(column=2, row=0, pady=8, padx=100,sticky=W,columnspan=2)
    infor = tk.Label(frame_2, text="Image Information:", font='bold').grid(column=0, row=0, pady=8, padx=30, sticky=W,columnspan=2)
    addr_1 = tk.Label(frame_2, text="Location: {}".format(filename)).grid(column= 0, row = 4,pady = 8,padx=30,sticky = W)
    h_2 = tk.Label(frame_2, text="Height: {}".format(h_1)).grid(column= 0, row = 2,pady = 8,padx=30,sticky = W)
    w_2 = tk.Label(frame_2, text="Width: {}".format(w_1)).grid(column= 0, row = 3,pady = 8,padx=30,sticky = W)

notelable = tk.Label(frame_2, text="Nhấn 'Choose File' để chọn ảnh", font='bold').grid(column=0, row=5, pady=8,padx=30, sticky=W)
chose_button = tk.Button(frame_2, text="Choose File",command=open).grid(column= 0, row = 6,pady=8,padx=30,sticky = W)


window.geometry("")
window.mainloop()


