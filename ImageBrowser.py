from tkinter import *
# open cv imaging
import cv2
from tkinter import filedialog

root = Tk()
root.title("Image Viewer")

def open():

    global my_image
    root.filename = filedialog.askopenfilename(initialdir="C:/Users", title="Select an Image",
                                               filetypes=(("png files", "*.png"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename)
    my_image = cv2.imshow(root.filename)
    my_image_label = Label(image=my_image).pack()

my_btn = Button(root, text="Open file", command=open).pack()

root.mainloop()