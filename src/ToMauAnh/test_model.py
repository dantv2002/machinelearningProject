import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

SIZE = 160
gray_img = []

img = cv2.imread('./images/3.jpg', 1)
img = cv2.resize(img, (SIZE,SIZE))
img = img.astype('float32') / 255.0
gray_img.append(img_to_array(img))

json_file = open('models/image_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
images_model = model_from_json(loaded_model_json)

images_model.load_weights("models/images_model_weights.h5") 
print("Loaded model from disk")

def plot_images(color,grayscale,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('Color Image', color = 'green', fontsize = 20)
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.title('Grayscale Image ', color = 'black', fontsize = 20)
    plt.imshow(grayscale)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)
   
    plt.show()

i = 0
predicted = np.clip(images_model.predict(gray_img[i].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
plot_images(gray_img[i],gray_img[i],predicted)
