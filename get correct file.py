import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image
from Numbers import numbers

def open_file(window):
    filepath = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select file",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*"))    )
    if not filepath:
        return

    try:
        img = Image.open(filepath)
        window.destroy()
        numbers(img)
    except Exception as e:
        print("Error opening File", e)

def main():
    window = tk.Tk()
    window.title("Get File")
    button = tk.Button(window, text="Upload", command=lambda: open_file(window))

    button.pack()
    window.mainloop()

if __name__ == "__main__":
    main()