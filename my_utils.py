import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_to_list(path_to_data):
    images = []
    class_no = []
    list_class = os.listdir(path_to_data)
    for x in range(0, len(list_class)):
        img_list = os.listdir(os.path.join(path_to_data, str(x)))
        print(x, end=' ')
        for y in img_list:
            img = cv2.imread(os.path.join(path_to_data, str(x), y))
            images.append(img)
            class_no.append(x)
    images = np.array(images)
    class_no = np.array(class_no)
    return images, class_no


def plot_training_data(no_class, x_train, y_train):
    num_of_samples = []
    for j in range(0, no_class):
        x_selected = x_train[y_train == j]
        num_of_samples.append(len(x_selected))

    # print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, no_class), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
