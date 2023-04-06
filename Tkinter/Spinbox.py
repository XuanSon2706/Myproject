import tkinter as tk
from tkinter import *
from tkinter import messagebox

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Chào bạn đã đến đây", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

spin_box = tk.Spinbox(window,from_= 0, to = 20, width =20).grid(column = 0 , row = 1)

kichthuoc = ""
window.geometry(kichthuoc)

window.mainloop()