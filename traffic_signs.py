from my_utils import data_to_list

# All image data into a single list
image_list, class_no = data_to_list(
    '/home/naseem/PycharmProjects/DetectTrafficSigns-ComputerVision-python/Data'
)
print(image_list, class_no)
