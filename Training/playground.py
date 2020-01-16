from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
import sys
sys.path.insert(1, '../Product/')
from addresses import Addresses
from addresses import load_image_for_prediction
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
import numpy as np

model = VGG16()

test = np.zeros(10)
test[:] = [1]*10
print(test)