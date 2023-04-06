import tkinter as tk
from tkinter import *
from tkinter import messagebox

window = Tk()
window.title ("Welcome my project")

def cmd():
    messagebox.showinfo("Welcome","Chào bạn")
    messagebox.showwarning('Warning', 'Virus coming!')
    messagebox.showerror('Error', 'Hello im vius')
button = tk.Button(window, text = "Open",command = cmd).pack()

kichthuoc = ""
window.geometry(kichthuoc)

window.mainloop()