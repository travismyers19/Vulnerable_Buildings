import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_filename = 'inception_model.h5'
data_folder = '/home/ubuntu/Insight/s3mnt/data2/preprocessed'
steps = 50
epochs = 1

model = load_model(model_filename)
print(model.summary())

training_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
training_generator = training_datagen.flow_from_directory(
    directory = data_folder,
    target_size = (299, 299),
    batch_size = 32,
    class_mode = 'categorical')

model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=epochs)

model.save(model_filename)