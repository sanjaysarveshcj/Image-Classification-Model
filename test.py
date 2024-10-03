import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
import os
import pickle

model = tf.keras.models.load_model('cifar10_model.h5')

def load_cifar10_label_names(data_dir):
    meta_file_path = os.path.join(data_dir, 'batches.meta')
    with open(meta_file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    return data['label_names']

data_dir = 'cifar-10-batches-py' 

label_names = load_cifar10_label_names(data_dir) 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)
    
    predicted_label = label_names[predicted_class[0]]
    
    return predicted_label

img_path = 'd.jpg' 

predicted_label = predict_image_class(img_path)
print(f"The image is classified as: {predicted_label}")

img = image.load_img(img_path, target_size=(32, 32))
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.show()
