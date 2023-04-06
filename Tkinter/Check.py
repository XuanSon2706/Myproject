import tkinter as tk
from tkinter import *
from tkinter import ttk

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Giới tính của bạn là:", font= ("Arial bold", 13))
label.grid(column = 0 , row = 0)

check = tk.BooleanVar()
check_button = tk.Checkbutton(window, text = "Nam",font= ("Arial bold", 13), var = check.set(False)).grid(column = 0 , row = 1)
check_button_1 = tk.Checkbutton(window, text = "Nữ",font= ("Arial bold", 13), var = check.set(False)).grid(column = 1 , row = 1)

kichthuoc = "500x200"
window.geometry(kichthuoc)

window.mainloop()