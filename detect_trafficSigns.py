import cv2
from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import img_to_array
import numpy as np

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
    success, img_og = cap.read()
    img_rgb = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (32, 32))
    img = img.astype('float32')/255
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0]
    predict = traffic_labels[prediction.argmax()]
    print(predict)

    cv2.imshow('Webcam', img_og)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
