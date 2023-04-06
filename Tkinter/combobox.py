import tkinter as tk
from tkinter import *
from tkinter import ttk

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Chào bạn đã đến đây", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

combo = ttk.Combobox(window)
combo["value"] = ("yellow","red","blue","pink")
combo.current(1)
combo.grid(column = 0 , row = 1)


kichthuoc = ""
window.geometry(kichthuoc)

window.mainloop()
