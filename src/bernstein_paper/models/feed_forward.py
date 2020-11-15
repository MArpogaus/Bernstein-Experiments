#!env python3
# -*- coding: utf-8 -*-
# AUTHOR INFORMATION ##########################################################
# file   : feed_forward.py
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
# modified time : 2020-05-16 13:34:12
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-03-31 19:22:59
#  changes made : newly written
###############################################################################

# PYTHON AUTHORSHIP INFORMATION ###############################################
# ref.: https://stackoverflow.com/questions/1523427

"""baseline.py: [Description]"""

__author__ = ["Marcel Arpogaus"]
# __authors__ = ["author_one", "author_two" ]
# __contact__ = "kontakt@htwg-konstanz.de"

# __copyright__ = ""
# __license__ = ""

__date__ = "2020-03-31 19:22:59"
# __status__ = ""
# __version__ = ""

# REQUIRED PYTHON MODULES #####################################################
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model


def build_model(history_shape,
                meta_shape,
                output_shape,
                hidden_layers=[
                    dict(units=50,
                         activation='elu',
                         kernel_initializer="he_normal")
                ],
                output_layer_kwds=dict(activation='linear'),
                batch_normalization=True,
                name=None):

    hist_in = Input(shape=history_shape)
    meta_in = Input(shape=meta_shape)

    x1 = Flatten()(hist_in)
    x2 = Flatten()(meta_in)

    x = Concatenate()([x1, x2])

    if batch_normalization:
        x = BatchNormalization()(x)

    for kwds in hidden_layers:
        x = Dense(**kwds)(x)
        if batch_normalization:
            x = BatchNormalization()(x)

    x = Dense(np.prod(output_shape), **output_layer_kwds)(x)
    x = Reshape(output_shape)(x)

    return Model(inputs=[hist_in, meta_in], outputs=x)
