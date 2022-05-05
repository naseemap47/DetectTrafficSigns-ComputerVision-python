import cv2
from keras.models import load_model
import pandas as pd

frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Model
model = load_model('/home/naseem/PycharmProjects/DetectTrafficSigns-ComputerVision-python/Model.h5')
labels = pd.read_csv('labels.csv')
# print(labels['Name'])
traffic_labels = labels['Name']

while True:
    success, img = cap.read()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
