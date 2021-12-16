""" This module is used to find coordinates of the phone in the image."""

# Loading dependencies.
import os
import sys

# Data pre-processing libraries.
import cv2
import numpy as np

# Machine learning modelling libraries.
from keras.models import load_model

def find_phone(path):
    """ This function is used to predict coordinates using the trained model. """
    # Getting file name and path from the input.
    
    print("path:",path)
    file_name = list(path.split(os.sep))[-1]
    path_list = list(path.split(os.sep))[:-1]
    print("filename,path_list",file_name,path_list)
    new_path = ''
    for i, element in enumerate(path_list):
        print("element", element)
        if i < len(path_list) - 1:
            new_path = new_path + element + os.sep
        else:
            new_path = new_path + element

    print("new_path: ",new_path)
    # Setting the path to folder containing weights.
    os.chdir(new_path)

    # Reading the image file.
    x_variable = []
    img = cv2.imread(file_name)
    resized_image = cv2.resize(img, (64, 64))
    x_variable.append(resized_image.tolist())
    x_variable = np.asarray(x_variable)
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))

    # Loading the Deep Learning Model.
    model = load_model('train_phone_finder_weights.h5')
    result = model.predict(x_variable)
    print("\n\nPhone in image {0} is located at x-y coordinates given below."
          .format(str(file_name)))
    print("\n{:.4f} {:.4f}".format(result[0][0], result[0][1]))
    image = cv2.circle(resized_image, (result[0][0], result[0][1]), radius=0, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("result.png",image)

def main():
    """ This function is used to run the program. """
    find_phone(sys.argv[1])

if __name__ == "__main__":
    main()
