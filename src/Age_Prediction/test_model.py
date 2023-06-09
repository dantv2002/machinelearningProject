import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir

# load model 
def ageModel():
    json_file = open('models/age_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    age_model = model_from_json(loaded_model_json)

    age_model.load_weights("models/age_model_weights.h5")

    return age_model

age_model = ageModel()
output_indexes = np.array([i for i in range(0, 101)])

# ------------------------FaceDetectorYN-----------------------
detector = cv2.FaceDetectorYN.create(
    'models/face_detection_yunet_2022mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

cap = cv2.VideoCapture(0)  # capture webcam
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])
thickness = 2

while (True):
    ret, img = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    img = cv2.resize(img, (frameWidth, frameHeight))
    input = img
    faces = detector.detect(img)

    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x = coords[0]
            y = coords[1]
            w = coords[2]
            h = coords[3]
            # Ve hinh chu nhat quanh mat
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle to main image

            # Crop mat
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            try:
                # Them magin
                margin = 30
                margin_x = int((w * margin) / 100);
                margin_y = int((h * margin) / 100)
                detected_face = img[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            except:
                print("detected face has no margin")

            try:
                # Dua mat vao mang predict
                detected_face = cv2.resize(detected_face, (224, 224))

                img_pixels = img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # Hien thi thong tin tuoi
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                # Ve khung thong tin
                info_box_color = (46, 200, 255)
                triangle_cnt = np.array(
                    [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
                cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                cv2.rectangle(img, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90), info_box_color,
                                cv2.FILLED)

                cv2.putText(img, apparent_age, (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)


            except Exception as e:
                print("exception", str(e))

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()