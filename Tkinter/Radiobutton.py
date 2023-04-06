import tkinter as tk
from tkinter import *
from tkinter import ttk

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Giới tính của bạn là:", font= ("Arial bold", 13))
label.grid(column = 0 , row = 0)

radio_1 = tk.Radiobutton(window,text = "Nam", value = 1).grid(column = 1 , row = 0)
radio_2 = tk.Radiobutton(window,text = "Nữ", value = 2).grid(column = 2 , row = 0)
radio_3 = tk.Radiobutton(window,text = "Khác", value = 3).grid(column = 3 , row = 0)

kichthuoc = ""
window.geometry(kichthuoc)

window.mainloop()