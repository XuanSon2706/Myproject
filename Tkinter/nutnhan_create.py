import tkinter as tk
from tkinter import *

window = Tk()
window.title ("Welcome my project")

label = tk.Label(window, text = "Chào bạn đã đến đây", font= ("Arial bold", 20))
label.grid(column = 0 , row = 0)

button = tk.Button(window, text = "Open")
button.grid(column = 0 , row = 1)

kichthuoc = "300x200"
window.geometry(kichthuoc)

window.mainloop()