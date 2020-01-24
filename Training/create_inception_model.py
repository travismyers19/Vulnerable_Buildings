import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model
from tensorflow.keras import metrics

model_filename = 'inception_model.h5'
unfrozen_layers = 9

model = InceptionV3(include_top=False, pooling='avg')
dense1 = Dense(1024, activation='relu')(model.outputs[0])
dense2 = Dense(512, activation='relu')(dense1)
dense3 = Dense(256, activation='relu')(dense2)
output = Dense(3, activation='softmax')(dense3)
model = Model(inputs=model.inputs, outputs=output)
for index in range(len(model.layers) - unfrozen_layers):
    model.layers[index].trainable = False
#print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])
model.save(model_filename)