import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import metrics
import horovod.keras as hvd

def create_model(model_filename, unfrozen_layers)
    hvd.init()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    opt = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size()))

    model = InceptionV3(include_top=False, pooling='avg')
    dense1 = Dense(1024, activation='relu')(model.outputs[0])
    dropout1 = Dropout(0.2)(dense1)
    norm1 = BatchNormalization()(dropout1)
    dense2 = Dense(512, activation='relu')(norm1)
    dropout2 = Dropout(0.2)(dense2)
    norm2 = BatchNormalization()(dropout2)
    dense3 = Dense(256, activation='relu')(norm2)
    dropout3 = Dropout(0.2)(dense3)
    norm3 = BatchNormalization()(dropout3)
    output = Dense(3, activation='softmax')(norm3)
    model = Model(inputs=model.inputs, outputs=output)
    for index in range(len(model.layers) - unfrozen_layers):
        model.layers[index].trainable = False
    #print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])
    model.save(model_filename)

if __name__ == '__main__':
    model_filename = 'inception_model2.h5'
    unfrozen_layers = 21
    create_model(model_filename, unfrozen_layers)