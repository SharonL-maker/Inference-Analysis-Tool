import numpy as np
import cv2
# import tensorflow as tf
import matplotlib.pyplot as plt
from ToolTip_Widget import CreateToolTip
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from myGenKeras import DataGen
import pandas as pd
import os
import time

########################### Window 1 - Initial Details ###########################

############ Functions ############



def open_file():

    filename = filedialog.askopenfilename(initialdir="C:/ML", title="Select a model",
                                               filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))

    model_var.set(filename)
    print(model_var.get())


def open_folder_Images():

    foldername_images = filedialog.askdirectory(initialdir="C:/ML", title="Select a Validation Images directory")

    trainX_var.set(foldername_images)
    print(trainX_var.get())

def open_folder_Masks():

    foldername_masks = filedialog.askdirectory(initialdir="C:/ML", title="Select a Validation Masks directory")
    trainY_var.set(foldername_masks)
    print(trainY_var.get())


root_1 = Tk()

model_var = StringVar()
trainX_var = StringVar()
trainY_var = StringVar()

root_1.geometry('300x200+1000+100')  # 1000 pixels fro left and 100 from top of the screen
root_1.title("Batch Inference Tool")
label_1 = Label(root_1, text="Welcome to batch Inference Analysis tool!").pack()

btn_1 = Button(root_1, text="Select a model file", command=open_file).pack()
btn_2 = Button(root_1, text="Select a Validation set Images folder", command=open_folder_Images()).pack()
btn_3 = Button(root_1, text="Select a Validation set Masks folder", command=open_folder_Masks()).pack()

label_2 = Label(root_1, text="To change the model or Test set,").pack()
label_3 = Label(root_1, text=" click on the 'Select' buttons").pack()
label_4 = Label(root_1, text="When Done, click on the 'Exit' button").pack()
# ToolTip_Widget.CreateToolTip(btn_1, text="TensorFlow model files, .h5 filetype")
# ToolTip_Widget.CreateToolTip(btn_2, text="A Validation folder with .jpg file or .bmp files")

root_1.mainloop()



########################### Batch Inference ###########################

"""
At first, the data is generated artificially and randomly,
just to demonstrate the possibility of handling the data representation properly
In addition, this part will include progressbar
"""

# num_of_images = np.random.randint(low=10, high=100, size=1)

# print("The integer sampled is: {}".format(num_of_images))
# print("The shape of the array is: {}".format(np.shape(num_of_images)))

# images = np.linspace(1, num_of_images, num_of_images)
# print("The size of the images array is: {}".format(images))
# print(images)

# probabilities = []
# for i in range(0, len(images)):
  #   probabilities.append(np.random.rand())

# print(probabilities)

# trainX_path = my_folder
# trainY_path = my_folder

root_prog = Tk()
root_prog.geometry('300x200+500+200')

progress_status = StringVar()
Label_prog_1 = Label(root_prog, textvariable=progress_status).pack()

# progress = Progressbar(root_prog, orient=HORIZONTAL, length=100, mode='determinate')


# def bar():
#     progress['value'] = 20
#     root_prog.update_idletasks()
#     time.sleep(1)
#
#     progress['value'] = 40
#     root_prog.update_idletasks()
#     time.sleep(1)
#
#     progress['value'] = 50
#     root_prog.update_idletasks()
#     time.sleep(1)
#
#     progress['value'] = 60
#     root_prog.update_idletasks()
#     time.sleep(1)
#
#     progress['value'] = 80
#     root_prog.update_idletasks()
#     time.sleep(1)
#     progress['value'] = 100


# progress.pack(pady=10)

# This button will initialize
# the progress bar
# Button(root_prog, text='Start', command=bar).pack(pady=10)


try:
    trainX_path = trainX_var.get()
    trainY_path = trainY_var.get()
    model_name = model_var.get()
except:
    trainX_path = r"C:\ML\ML\Data\Val\images"
    trainY_path = r"C:\ML\ML\Data\Val\masks"
    model_name = r"C:\ML\ML\model19.h5"

# trainX_path = r"C:\Users\USER\PycharmProjects\AutoEncoder1\images\data"
# trainY_path = r"C:\Users\USER\PycharmProjects\AutoEncoder1\masks\data"

onlyfiles = next(os.walk(trainX_path))[2]
num_of_files = len(onlyfiles)

# Creating the division of the progressbar with the current number of files
div_gen = num_of_files/100
div_1 = int(np.ceil(div_gen*20))
div_2 = int(np.ceil(div_gen*40))
div_3 = int(np.ceil(div_gen*60))
div_4 = int(np.ceil(div_gen*80))

# Creating file námes' list
file_name_tooltip = []

train_ids=[]
for root, dirs, files in os.walk(trainX_path):
    for filename in files:
        print(filename)
        train_ids.append(filename)

crop = None#(1624,150,624,1048)

batch_size = 1
image_size = 128


gen = DataGen(train_ids, trainX_path, trainY_path,Aug=False, crop=crop, batch_size=batch_size, image_size=image_size)





def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.3):

    out = img.copy()
    img_layer = img.copy()
    M = cv2.threshold(mask,0.33,1,cv2.THRESH_BINARY)
    img_layer[M[1]==1] = [0, 255, 255]
    M = cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
    img_layer[M[1]==1] = [255, 0, 0]
    M = cv2.threshold(mask,0.75,1,cv2.THRESH_BINARY)
    img_layer[M[1]==1] = [0, 255, 0]
    M = cv2.threshold(mask,0.95,1,cv2.THRESH_BINARY)
    img_layer[M[1]==1] = [0, 0, 255]
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)




runInference = True
if runInference:

    import keras


    segModel1 = keras.models.load_model(model_name)#, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})#('seg_256_RLR_1.h5'),seg_256_RLR_DiceMetric,seg_256_RLR_DiceMetric2,seg_128_RLR_DiceMetric3
    segModel1.summary()
    ####

    smooth = 1.


    def dice_coef(y_true, y_pred):  #############NOT WORKING IN KERAS :/
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = keras.backend.sum(keras.backend.abs(y_true_f * y_pred_f), axis=-1)
        return (2. * intersection + smooth) / (
                    keras.backend.sum(keras.backend.square(y_true_f), -1) + keras.backend.sum(
                keras.backend.square(y_pred_f), -1) + smooth)


    def dice_coef_loss(y_true, y_pred):
        y_pred_f = keras.backend.flatten(y_pred)
        reg = keras.backend.sum(keras.backend.abs(y_pred_f), axis=-1)
        lamda = 1 / (image_size * image_size)
        loss = 1.0 - dice_coef(y_true, y_pred) + 0.0 * lamda * reg
        return keras.backend.log(loss)


    # opt = keras.optimizers.RMSprop(lr=0.00003,decay=0.0005)

    # keras.optimizers.M
    opt = keras.optimizers.Adam(lr=1e-5, decay=0.00005)
    segModel1.compile(loss=[dice_coef_loss], optimizer=opt,
                     metrics=[dice_coef])  # ,metrics=[dice_coef])'binary_crossentropy'
    ####
    # segModel2 = keras.models.load_model('kerasReg1DiceCoeff128.h5')
    # segModel2.summary()
    index = range(0, train_ids.__len__())
    index = np.random.permutation(index)
    counter = 0
    L_array = []
    timing_arr = []
    cv2.namedWindow("Segmentation Viewer - Online", flags=cv2.WINDOW_FREERATIO)

    # Save Path

    images = []
    # cv2.namedWindow("seg2", flags=cv2.WINDOW_AUTOSIZE)
    # while (counter<=44):
    for file in os.listdir(trainX_path):
        x, y = gen.__getitem__(index[counter])
        im = x[0]
        # im = cv2.resize(im,(624,1048))

        p1 = segModel1.predict(x)
        startTime1 = time.time()
        l1 = segModel1.evaluate(x, y)


        endTime1 = time.time()
        L_array.append(l1[1])
        print(L_array[counter])
        timing_arr.append(1000*(endTime1 - startTime1))
        # startTime2 = time.time()
        # p2 = segModel2.predict(x)
        # endTime2 = time.time()
        rgb = cv2.imread(trainX_path + '/' + train_ids[index[counter]], cv2.IMREAD_ANYCOLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BAYER_BG2BGR)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        zeros = np.zeros((rgb.shape[0],rgb.shape[1]),dtype=np.uint8)
        if crop is not None:
            rgb=rgb[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]
            masked1 = mask_color_img(rgb, cv2.resize(p1[0], (624, 1048)))
        else:
            masked1 = mask_color_img(rgb, cv2.resize(p1[0], (rgb.shape[1], rgb.shape[0])))
        # masked2 = mask_color_img(rgb,cv2.resize(p2[0],(624,1048)))
        cv2.putText(masked1, '{:4.1f}'.format(1000*(endTime1 -startTime1))+' ms', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255))
        cv2.putText(masked1, '{:4.3f}'.format(L_array[counter]), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255))

        cv2.imshow("Segmentation Viewer - Online", masked1)
        # Creating a current name for image saving

        # Saving the current image
        # cv2.putText(masked2,'{:4.1f}'.format(1000*(endTime2 -startTime2)),(10,500),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
        # cv2.imshow("seg2", masked2)



        cv2.waitKey(10)

        counter = counter+1
        images.append(counter)
        file_name_tooltip.append(file)
        print("Image number: {}, out of {}".format(counter, num_of_files))

        progress_status.set(str(int(np.ceil((counter/num_of_files)*100))) + " %")
        # if (counter > div_1 & counter < div_2):
        #     progress['value'] = 20
        #     root_prog.update_idletasks()
        # elif (counter > div_2 & counter < div_3):
        #     progress['value'] = 40
        #     root_prog.update_idletasks()
        # elif (counter > div_3 & counter < div_4):
        #     progress['value'] = 60
        #     root_prog.update_idletasks()
        # elif (counter > div_4 & counter < num_of_files):
        #     progress['value'] = 80
        #     root_prog.update_idletasks()
        # else:
        #     progress['value'] = 100
        #     root_prog.update_idletasks()
        #     root_prog.quit()

    root_prog.mainloop()
########################### Analysis ###########################

############ Choosing a percentage window ############

perc = 0.25

probabilities = L_array

# def load_perc():
#
#     input_perc = entry_1.widget.get()
#     print(input_perc)
#     output_perc = "The chosen percentage is: "+str(int(100*input_perc))
#     var_1.set(output_perc)


root_2 = Tk()
var_1 = StringVar()


root_2.geometry('300x200+1000+100') # 1000 pixels fro left and 100 from top of the screen
root_2.title("Batch Inference Tool")
label_4 = Label(root_2, text="Please enter a desired percentage to observe (0-1)").pack()
entry_1 = Entry(root_2, textvariable=var_1, justify=CENTER).pack()
# entry_1.focus_force()
# entry_1.bind("<Return>", on_change())


def getvalue():

    print(str(var_1.get()))
    global perci
    perci = float(str(var_1.get()))
    root_2.quit()


btn_3 = Button(root_2, text="Load percentage" ,command=getvalue).pack()
label_5 = Label(root_2, text="After clicking, close the window").pack()
label_6 = Label(root_2, text="To move to score table, close the graph window").pack()
# label_5 = Label(root_2, textvariable=var_1).pack()

root_2.mainloop()


############ Display of results ############

# Finding the values below the chosen percentage

perc = float(perci)
print("The chosen percentage is: {}".format(str(perc)))
min_probabilities = np.min(probabilities)
max_probabilities = np.max(probabilities)
threshold_value = min_probabilities + (max_probabilities - min_probabilities)*np.float(perc)
average_probability = np.mean(probabilities)
out_of_range = []
avg_vec = []
for i in range(0, len(probabilities)):

    if (probabilities[i] < threshold_value):
        out_of_range.append(probabilities[i])
    else:
        out_of_range.append(0)
    avg_vec.append(average_probability)

# image_number = int(np.linspace(0, len(images)))
out_base_arr = []
for i in range(0, len(out_of_range)):
    out_base_arr.append(i)
plt.figure()
plt.title("Results, Metrics vs. Inferences")
plt.xlabel("Image Inference")
plt.ylabel("Metrics Score")
plt.plot(images, probabilities, 'bo', out_base_arr, out_of_range, 'ro', images, avg_vec, 'g-')
plt.grid()
plt.legend(("Probabilities of all images", "Probabilities of outliers"), loc="lower right")
plt.show()

############ Scores presentation ############

"""This part uses the ToolTip to present the image data.
The ToolTip"""

# ToolTip collection creation



num_cols = 3
num_rows = int(np.ceil(len(probabilities) / num_cols))

root_3 = Tk()
root_3.title("Probability analysis")

main_label1 = Label(root_3, text="Highlighting the {}%".format((perc) * 100), font=("Arial Bold", 20))
main_label1.grid(column=int(num_cols / 2), row=0)
main_label2 = Label(root_3, text=" Lowest IOU scores", font=("Arial Bold", 20))
main_label2.grid(column=int(num_cols / 2), row=1)

for each_row in range(0, num_rows):
    for each_col in range(0, num_cols):
        i = each_row + each_col
        if probabilities[i] in out_of_range:
            lbl_i = Label(root_3, text=probabilities[i], background="Orange", font=("Arial Bold", 16))
            lbl_i.grid(column=each_col, row=each_row + 2)
            CreateToolTip(lbl_i, file_name_tooltip[i])
            lbl_i = Label(root_3, text=probabilities[i], font=("Arial Bold", 16))
            lbl_i.grid(column=each_col, row=each_row + 2)
            CreateToolTip(lbl_i, file_name_tooltip[i])

root_3.mainloop()
