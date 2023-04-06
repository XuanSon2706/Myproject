import tkinter as tk
from tkinter import *

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Chào bạn đã đến đây", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

window.mainloop()
