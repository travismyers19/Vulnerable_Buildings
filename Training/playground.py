from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
import sys
from addresses import Addresses
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
import numpy as np
import os
from buildingclassifier import BuildingClassifier


image_classifier = BuildingClassifier('/home/ubuntu/Insight/s3mnt/Models/Baseline/inception_model_trained_6-10.h5')
metrics = image_classifier.get_ternary_model_statistics('/home/ubuntu/Insight/s3mnt/Test')
print(metrics)