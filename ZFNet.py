#https://github.com/rainer85ah/Papers2Code/blob/master/ZFNet/ZFNet.py
# -*- coding: utf-8 -*-
import keras.backend as K
from keras import layers, optimizers
from keras.models import Model
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, \
    ZeroPadding2D, Input

__author__ = 'Rainer Arencibia'
__copyright__ = 'Copyright 2017, Rainer Arencibia'
__credits__ = 'ZFNet Team, Keras Team and TensorFlow Team'
__license__ = 'MIT'
__version__ = '1.00'
__status__ = 'Prototype'


class ZFNet:

    def __init__(self):

        self.model = None

    def build(self, input_shape=None, num_outputs=1000):
        """
                Args:
                    input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels) TensorFlow Format!!
                    num_outputs: The number of outputs at final softmax layer
                Returns:
                    A compile Keras model.
                """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple like (nb_rows, nb_cols, nb_channels)")

        # (227, 227, 3)
        input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=197,
                                          data_format=K.image_data_format(), include_top=True)
        img_input = Input(shape=input_shape)
        # x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(96, (7, 7), strides=(2, 2), name='conv1')(img_input)
        # (55, 55, 96)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)
        # (27, 27, 96)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)

        x = Conv2D(256, (5, 5), strides=(4, 4), name='conv2')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(x)
        x = BatchNormalization(axis=3, name='bn_conv2')(x)

        x = Conv2D(512, (3, 3), strides=(1, 1), padding=1, name='conv3')(x)
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding=1, name='conv4')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding=1, name='conv5')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

        x = Dense(units=4096)(x)
        x = Dense(units=4096)(x)
        x = Dense(units=num_outputs)(x)
        x = Activation('softmax')(x)

        self.model = Model(inputs=img_input, outputs=x, name='ZFNet Model')
        return self.model

    def compile(self, optimizer='sgd'):

        optimizer_dicc = {'sgd': optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                          'rmsprop': optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                          'adagrad': optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
                          'adadelta': optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                          'adam': optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)}

        self.model.compile(optimizer=optimizer_dicc[optimizer], loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def predict(self, image_batch):
        import numpy as np

        predictions = []
        for image in image_batch:
            pred = self.model.predict(image)
            predictions.append(pred)

        return np.asarray(predictions)

if __name__ == '__main__':

    pass

# Assembling always help. Train 7 different models and average the predictions or outputs(probabilities) => Increase the
# Accuracy in 2%.

# Note: smaller filter and more filters(densely) =>