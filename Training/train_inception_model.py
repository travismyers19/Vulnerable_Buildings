import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import horovod.tensorflow.keras as hvd
import os

def train_model(model_filename, model_output_filename, epochs, batch_size, data_folder):
    hvd.init()
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    model = load_model(model_filename)
    print(model.summary())

    datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)
    training_generator = datagen.flow_from_directory(
        directory = data_folder,
        target_size = (299, 299),
        batch_size = batch_size,
        class_mode = 'categorical',
        subset = 'training')

    validatn_generator = datagen.flow_from_directory(
        directory = data_folder,
        target_size = (299, 299),
        batch_size = 1,
        class_mode = 'categorical',
        subset = 'validation')

    history = model.fit_generator(callbacks=callbacks, generator=training_generator, steps_per_epoch=training_generator.n//batch_size, epochs=epochs, validation_data=validatn_generator, validation_steps=validatn_generator.n)
    training_loss = np.array(history.history['loss'])
    training_accy = np.array(history.history['categorical_accuracy'])
    validatn_loss = np.array(history.history['val_loss'])
    validatn_accy = np.array(history.history['val_categorical_accuracy'])

    np.save('training_loss2.npy', training_loss)
    np.save('validatn_loss2.npy', validatn_loss)
    np.save('training_accy2.npy', training_accy)
    np.save('validatn_accy2.npy', validatn_accy)
    model.save(model_output_filename)

if __name__ == '__main__':
    model_filename = os.getenv('MODEL')
    model_output_filename = os.getenv('TRAINED_MODEL')
    epochs = int(os.getenv('EPOCHS'))
    batch_size = int(os.getenv('BATCH_SIZE'))
    data_folder = os.getenv('DATA_FOLDER')

    train_model(model_filename, model_output_filename, epochs, batch_size, data_folder)