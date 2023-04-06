import tkinter as tk
from tkinter import *

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Chào bạn đã đến đây", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

def cmd():
    label_2 = tk.Label(window, text = "thấy ảo không ^_^?", font= ("Arial bold", 20))
    label_2.grid(column=0, row=2)

button = tk.Button(window, text = "Open",command = cmd)
button.grid(column = 0 , row = 1)

kichthuoc = "300x200"
window.geometry(kichthuoc)

window.mainloop()