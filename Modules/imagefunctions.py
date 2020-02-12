import requests
import numpy as np
import csv
from PIL import Image
import io
from tensorflow.keras import preprocessing

def write_image_to_file(image, full_filename):
    np_image = Image.open(io.BytesIO(image))
    np_image = np.asarray(np_image)

    with open(full_filename, 'wb') as jpg_file:
        jpg_file.write(image)

def load_image_for_prediction(full_filename, target_size=(299, 299)):
    image = preprocessing.image.load_img(full_filename, target_size=target_size)
    return np.expand_dims(preprocessing.image.img_to_array(image), axis=0)/255.