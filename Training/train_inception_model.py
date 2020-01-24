import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

model_filename = 'inception_model.h5'
data_folder = '/home/ubuntu/Insight/s3mnt/data2/preprocessed'
steps = 10
epochs = 20

model = load_model(model_filename)
print(model.summary())

datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)
training_generator = datagen.flow_from_directory(
    directory = data_folder,
    target_size = (299, 299),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training')
validatn_generator = datagen.flow_from_directory(
    directory = data_folder,
    target_size = (299, 299),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation')

training_loss = np.zeros(epochs)
validatn_loss = np.zeros(epochs)
training_accy = np.zeros(epochs)
validatn_accy = np.zeros(epochs)

for epoch in range(epochs):
    model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=1)
    result = model.evaluate_generator(training_generator)
    training_loss[epoch] = result[0]
    training_accy[epoch] = result[1]
    result = model.evaluate_generator(validatn_generator)
    validatn_loss[epoch] = result[0]
    validatn_accy[epoch] = result[1]
    print(model.metrics_names)

np.save('training_loss.npy', training_loss)
np.save('validatn_loss.npy', validatn_loss)
np.save('training_accy.npy', training_accy)
np.save('validatn_accy.npy', validatn_accy)
model.save(model_filename)