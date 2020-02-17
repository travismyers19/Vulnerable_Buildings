import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import metrics
import horovod.keras as hvd
from tensorflow.keras.models import load_model
import os
from Modules.imagefunctions import load_image_for_prediction
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from Modules import customlosses

class BuildingClassifier:
    def __init__(self, model_filename, model2_filename=None):
        #If binary classifier, model_filename = good vs bad model, model2_filename = soft vs non-soft model
        self.model_filename = model_filename
        self.model2_filename = model2_filename
        self.model = None
        self.model2 = None

    def load_model(self):
        if self.model is None:
            self.model = load_model(self.model_filename, custom_objects={'focal_binary_crossentropy': customlosses.focal_binary_crossentropy, 'focal_categorical_crossentropy': customlosses.focal_categorical_crossentropy})
        if self.model2 is None and self.model2_filename is not None:
            self.model2 = load_model(self.model2_filename, custom_objects={'focal_binary_crossentropy': customlosses.focal_binary_crossentropy, 'focal_categorical_crossentropy': customlosses.focal_categorical_crossentropy})

    def create_inception_model(self, number_categories, dense_layer_sizes, dropout_fraction, unfrozen_layers, focal_loss=False):
        hvd.init()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        opt = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(learning_rate=0.001*hvd.size()))
        model = InceptionV3(include_top=False, pooling='avg')
        output = model.outputs[0]

        for layer_size in dense_layer_sizes:
            dense = Dense(layer_size, activation='relu')(output)
            dropout = Dropout(dropout_fraction)(dense)
            output = BatchNormalization()(dropout)
        
        if number_categories == 1:
            output = Dense(1, activation='sigmoid')(output)
        else:
            output = Dense(number_categories, activation='softmax')(output)
        model = Model(inputs=model.inputs, outputs=output)

        for index in range(len(model.layers) - unfrozen_layers):
            model.layers[index].trainable = False

        if number_categories == 1:
            the_metrics = [metrics.binary_accuracy]
            if focal_loss:
                loss = customlosses.focal_binary_crossentropy
            else:
                loss = 'binary_crossentropy'
        else:
            the_metrics = [metrics.categorical_accuracy]
            if focal_loss:
                loss = customlosses.focal_categorical_crossentropy
            else:
                loss = 'categorical_crossentropy'

        model.compile(optimizer=opt, loss=loss, metrics=the_metrics)
        model.save(self.model_filename)
        self.model = model

    def train_model(self, epochs, batch_size, training_directory, test_directory, trained_model_filename, metrics_filename, binary=False, weights=None):
        # The metrics will be saved as a numpy array:
        # First row: training accuracy
        # Second row: validation (test) accuracy
        # Third row: training loss
        # Fourth row: validation(test) loss
        if binary:
            class_mode = 'binary'
        else:
            class_mode = 'categorical'

        hvd.init()
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
        self.load_model()
        training_datagen = ImageDataGenerator(rescale=1./255., horizontal_flip=True)
        training_generator = training_datagen.flow_from_directory(
            directory = training_directory,
            target_size = (299, 299),
            batch_size = batch_size,
            class_mode = class_mode)
        test_datagen = ImageDataGenerator(rescale=1./255., horizontal_flip=False)
        validation_test_generator = test_datagen.flow_from_directory(
            directory = test_directory,
            target_size = (299, 299),
            batch_size = 1,
            class_mode = class_mode)

        metrics = np.zeros((2, epochs))

        for epoch in range(epochs):
            self.model.fit_generator(generator=training_generator, callbacks=callbacks, steps_per_epoch=training_generator.n//batch_size, class_weight=weights)
            result = self.model.evaluate_generator(validation_test_generator)
            metrics[0, epoch] = result[1]
            metrics[1, epoch] = result[0]
            print('Validation Accuracy: ' + str(metrics[0, epoch]))
            print('Validation Loss: ' + str(metrics[1, epoch]))

        self.model.save(trained_model_filename)
        np.save(metrics_filename, metrics)
        self.model_filename = trained_model_filename

    def classify_image(self, image):
        self.load_model()
        if self.model2 is None:
            return np.argmax(self.model.predict(image))
        prediction = self.model.predict(image)
        if prediction > 0.5:
            prediction = self.model2.predict(image)
            if prediction > 0.5:
                return 2
            return 1
        return 0

    def is_no_image(self, image):
        # Determines if the image is just "sorry there is no image at this location"
        return (image[0,50,50,0] > 0.8941176 and image[0,50,50,0] < 0.8941177)

    def get_statistics(self, test_directory):
        if self.model2_filename is None:
            return self.get_ternary_model_statistics(test_directory)
        return self.get_binary_model_statistics(test_directory)

    def get_ternary_model_statistics(self, test_directory):
        #Returns:  [accuracy, soft_precision, soft_recall, good_precision, good_recall]
        #soft = soft-story building, good = not a bad image
        #Assumes folder structure of test directory is bad_images, soft_story_images, non_soft_story_images

        model = load_model(self.model_filename, custom_objects={'focal_binary_crossentropy': customlosses.focal_binary_crossentropy, 'focal_categorical_crossentropy': customlosses.focal_categorical_crossentropy})
        total_number = 0
        total_correct = 0
        soft_true_positive = 0
        soft_false_positive = 0
        soft_false_negative = 0
        good_true_positive = 0
        good_false_positive = 0
        good_false_negative = 0

        directory = os.path.join(test_directory, 'soft_story_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = np.argmax(model.predict(image))
                if prediction == 0:
                    soft_false_negative += 1
                    good_false_negative += 1
                if prediction == 1:
                    soft_false_negative += 1
                    good_true_positive += 1
                if prediction == 2:
                    soft_true_positive += 1
                    good_true_positive += 1
                    total_correct += 1

        directory = os.path.join(test_directory, 'non_soft_story_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = np.argmax(model.predict(image))
                if prediction == 0:
                    good_false_negative += 1
                if prediction == 1:
                    good_true_positive += 1
                    total_correct += 1
                if prediction == 2:
                    soft_false_positive += 1
                    good_true_positive += 1

        directory = os.path.join(test_directory, 'bad_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = np.argmax(model.predict(image))
                if prediction == 0:
                    total_correct += 1
                if prediction == 1:
                    good_false_positive += 1
                if prediction == 2:
                    soft_false_positive += 1
                    good_false_positive += 1

        accuracy = total_correct/total_number
        soft_precision = soft_true_positive/(soft_true_positive + soft_false_positive)
        soft_recall = soft_true_positive/(soft_true_positive + soft_false_negative)
        good_precision = good_true_positive/(good_true_positive + good_false_positive)
        good_recall = good_true_positive/(good_true_positive + good_false_negative)
        return [accuracy, soft_precision, soft_recall, good_precision, good_recall]

    def get_binary_model_statistics(self, test_directory):
        #The model stored in self.model_filename is the model that distinguishes good from bad
        #self.model2_filename stores the model that distinguishes good from bad
        #Returns:  [accuracy, soft_precision, soft_recall, good_precision, good_recall]
        #soft = soft-story building, good = not a bad image
        #Assumes folder structure of test directory is bad_images, soft_story_images, non_soft_story_images

        good_model = load_model(self.model_filename, custom_objects={'focal_binary_crossentropy': customlosses.focal_binary_crossentropy, 'focal_categorical_crossentropy': customlosses.focal_categorical_crossentropy})
        soft_model = load_model(self.model2_filename, custom_objects={'focal_binary_crossentropy': customlosses.focal_binary_crossentropy, 'focal_categorical_crossentropy': customlosses.focal_categorical_crossentropy})
        total_number = 0
        total_correct = 0
        soft_true_positive = 0
        soft_false_positive = 0
        soft_false_negative = 0
        good_true_positive = 0
        good_false_positive = 0
        good_false_negative = 0

        directory = os.path.join(test_directory, 'soft_story_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = good_model.predict(image)
                if prediction > 0.5:
                    good_true_positive += 1
                    prediction = soft_model.predict(image)
                    if prediction > 0.5:
                        soft_true_positive += 1
                        total_correct += 1
                    else:
                        soft_false_negative += 1
                else:
                    good_false_negative += 1
                    soft_false_negative += 1

        directory = os.path.join(test_directory, 'non_soft_story_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = good_model.predict(image)
                if prediction > 0.5:
                    good_true_positive += 1
                    prediction = soft_model.predict(image)
                    if prediction > 0.5:
                        soft_false_positive += 1
                    else:
                        total_correct += 1
                else:
                    good_false_negative += 1

        directory = os.path.join(test_directory, 'bad_images')

        for filename in os.listdir(directory):
            if filename.endswith('.jpg'):
                total_number += 1
                full_filename = os.path.join(directory, filename)
                image = load_image_for_prediction(full_filename)
                prediction = good_model.predict(image)
                if prediction > 0.5:
                    good_false_positive += 1
                    prediction = soft_model.predict(image)
                    if prediction > 0.5:
                        soft_false_positive += 1
                else:
                    total_correct += 1

        accuracy = total_correct/total_number
        soft_precision = soft_true_positive/(soft_true_positive + soft_false_positive)
        soft_recall = soft_true_positive/(soft_true_positive + soft_false_negative)
        good_precision = good_true_positive/(good_true_positive + good_false_positive)
        good_recall = good_true_positive/(good_true_positive + good_false_negative)
        return [accuracy, soft_precision, soft_recall, good_precision, good_recall]
