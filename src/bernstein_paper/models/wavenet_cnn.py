#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : wavenet_cnn.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-30 13:38:39
# changed : 2020-11-17 16:18:57
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/)
#
# COPYRIGHT ###################################################################
# Copyright 2020 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization


def build_model(history_shape,
                meta_shape,
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

    hist_in = Input(shape=history_shape)
    meta_in = Input(shape=meta_shape)

    hist_conv = hist_in
    for kwds in conv_layers:
        hist_conv = Conv1D(**kwds)(hist_conv)

    x1 = Flatten()(hist_conv)
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

    return Model(inputs=[hist_in, meta_in], outputs=x, name=name)
