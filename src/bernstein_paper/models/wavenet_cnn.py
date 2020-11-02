#!env python3
# -*- coding: utf-8 -*-
# AUTHOR INFORMATION ##########################################################
# file   : wavenet_cnn.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-03-31 19:22:59
# COPYRIGHT ###################################################################
# NEEDS TO BE DISCUSSED WHEN RELEASED!
#
# PROJECT DESCRIPTION #########################################################
#
# NOTE: this project is following the PEP8 style guide
#
# Bla bla...
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-05-16 13:34:55
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-03-31 19:22:59
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import BatchNormalization


def build_model(input_shape,
                output_shape,
                conv_layers=[
                    dict(filters=20, kernel_size=2,
                         padding="causal",
                         activation="relu",
                         dilation_rate=rate)
                    for rate in (1, 2, 4, 8, 16, 32, 64)
                ],
                hidden_layers=[
                    dict(units=neurons,
                         activation='elu',
                         kernel_initializer="he_normal")
                    for neurons in (100, 100, 50)
                ],
                output_layer_kwds=dict(activation='linear'),
                batch_normalization=True,
                name=None):

    model = Sequential(name=name or 'wavenet')

    model.add(InputLayer(input_shape=input_shape))

    for kwds in conv_layers:
        model.add(Conv1D(**kwds))

    if batch_normalization:
        model.add(BatchNormalization())

    model.add(Flatten())

    for kwds in hidden_layers:
        model.add(Dense(**kwds))
        if batch_normalization:
            model.add(BatchNormalization())

    model.add(Dense(np.prod(output_shape), **output_layer_kwds))
    model.add(Reshape(output_shape))
    return model
