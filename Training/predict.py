from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

image_filename = '/home/ubuntu/Insight/s3mnt/data2/preprocessed/non_soft_story/87.jpg'
model_filename = 'inception_model.h5'

test_image = np.expand_dims(image.img_to_array(image.load_img(image_filename)), axis=0)
model = load_model(model_filename)

result = model.predict(test_image)
print(result)
