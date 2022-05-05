import os
import cv2
import numpy as np


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
