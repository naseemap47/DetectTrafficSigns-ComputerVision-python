from my_utils import data_to_list
from sklearn.model_selection import train_test_split

# All image data into a single list
image_list, class_no = data_to_list(
    '/home/naseem/PycharmProjects/DetectTrafficSigns-ComputerVision-python/Data'
)
# print(image_list, class_no)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(image_list, class_no, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

image_dimension = (32, 32, 3)
print("Data Shapes")
print("Train", end="")
print(x_train.shape, y_train.shape)
print("Validation", end="")
print(x_val.shape, y_val.shape)
print("Test", end="")
print(x_test.shape, y_test.shape)
assert (x_train.shape[0] == y_train.shape[
    0]), "The number of images in not equal to the number of lables in training set"
assert (x_val.shape[0] == y_val.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert (x_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (x_train.shape[1:] == image_dimension), " The dimesions of the Training images are wrong "
assert (x_val.shape[1:] == image_dimension), " The dimesionas of the Validation images are wrong "
assert (x_test.shape[1:] == image_dimension), " The dimesionas of the Test images are wrong"
