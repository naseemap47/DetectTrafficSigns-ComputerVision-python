from my_utils import data_to_list
from sklearn.model_selection import train_test_split
from my_utils import plot_training_data

# Switches
PLOT = True

# All image data into a single list
image_list, class_no = data_to_list(
    '/home/naseem/PycharmProjects/DetectTrafficSigns-ComputerVision-python/Data'
)
# print(image_list, class_no)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(image_list, class_no, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Bar Plot
if PLOT:
    plot_training_data(
        no_class=43,
        x_train=x_train,
        y_train=y_train
    )
