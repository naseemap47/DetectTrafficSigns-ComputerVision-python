import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


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


def create_generators(batch_size, no_class,
                      x_train, y_train,
                      x_val, y_val,
                      x_test, y_test):
    # to_categorical
    y_train = to_categorical(y_train, no_class)
    y_val = to_categorical(y_val, no_class)
    y_test = to_categorical(y_test, no_class)

    # Preprocessor
    train_preprocessor = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator(rescale=1/255)
    test_preprocessor = ImageDataGenerator(rescale=1/255)
    train_generators = train_preprocessor.flow(
        x_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    val_generators = val_preprocessor.flow(
        x_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    test_generators = test_preprocessor.flow(
        x_test, y_test,
        batch_size=batch_size,
        shuffle=False
    )
    return train_generators, val_generators, test_generators
