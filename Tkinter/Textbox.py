import tkinter as tk
from tkinter import *

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Nhập vào dòng chữ gì đó", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

txt = tk.Entry(window,width = 40)
txt.grid(column=0 , row = 1)

kichthuoc = ""
window.geometry(kichthuoc)

window.mainloop()
