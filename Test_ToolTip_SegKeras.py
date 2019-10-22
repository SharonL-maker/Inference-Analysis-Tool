import numpy as np
import cv2
# import tensorflow as tf
import matplotlib.pyplot as plt
import ToolTip_Widget
from tkinter import *
from tkinter import filedialog

########################### Window 1 - Initial Details ###########################

############ Functions ############

def open_file():

    global my_model
    filename = filedialog.askopenfilename(initialdir="C:/Users", title="Select a model",
                                               filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))

    my_model = filename
    print(my_model)
    return my_model


def open_folder():

    global my_folder
    foldername = filedialog.askdirectory(initialdir="C:/Users", title="Select a Validation Folder")

    my_folder = foldername
    print(my_folder)
    return my_folder


root_1 = Tk()
root_1.geometry('300x200+1000+100') # 1000 pixels fro left and 100 from top of the screen
root_1.title("Batch Inference Tool")
label_1 = Label(root_1, text="Welcome to batch Inference Analysis tool!").pack()

btn_1 = Button(root_1, text="Select a model file", command=open_file).pack()
btn_2 = Button(root_1, text="Select a Validation set folder", command=open_folder).pack()

label_2 = Label(root_1, text="To change the model or Validation,").pack()
label_3 = Label(root_1, text=" click on the 'Select' buttons").pack()
# ToolTip_Widget.CreateToolTip(btn_1, text="TensorFlow model files, .h5 filetype")
# ToolTip_Widget.CreateToolTip(btn_2, text="A Validation folder with .jpg file or .bmp files")

root_1.mainloop()


########################### Batch Inference ###########################

"""
At first, the data is generated artificially and randomly,
just to demonstrate the possibility of handling the data representation properly
In addition, this part will include progressbar
"""

num_of_images = np.random.randint(low=10, high=100, size=1)

print("The integer sampled is: {}".format(num_of_images))
print("The shape of the array is: {}".format(np.shape(num_of_images)))

images = np.linspace(1, num_of_images, num_of_images)
print("The size of the images array is: {}".format(images))
print(images)

probabilities = []
for i in range(0, len(images)):
    probabilities.append(np.random.rand())

print(probabilities)
########################### Analysis ###########################

############ Choosing a percentage window ############

perc = 0

def load_perc():

    input_perc = entry_1.get()
    print(input_perc)
    output_perc = "The chosen percentage is: "+str(int(100*input_perc))
    var_1.set(output_perc)


root_2 = Tk()
var_1 = StringVar()
root_2.geometry('300x200+1000+100') # 1000 pixels fro left and 100 from top of the screen
root_2.title("Batch Inference Tool")
label_4 = Label(root_2, text="Please enter a desired percentage to observe (0-1)").pack()
entry_1 = Entry(root_2, justify=CENTER).pack()
# entry_1.focus_force()
btn_3 = Button(root_2, text="Load percentage",command=load_perc).pack()
label_5 = Label(root_2, textvariable=var_1).pack()

root_2.mainloop()


print("The chosen percentage is: {}".format(perc))
############ Display of results ############

# Finding the values below the chosen percentage

min_probabilities = np.min(probabilities)
max_probabilities = np.max(probabilities)
threshold_value = min_probabilities + (max_probabilities - min_probabilities)*perc
out_of_range = []
for i in range(0, len(probabilities)):

    if (probabilities[i] < threshold_value):
        out_of_range.append(probabilities[i])

# image_number = int(np.linspace(0, len(images)))

plt.figure()
plt.plot(images, probabilities, 'b0', [n for n in range(0, len(out_of_range))], out_of_range, 'r')
plt.grid()
plt.show()
