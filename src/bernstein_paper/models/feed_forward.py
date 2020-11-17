#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : feed_forward.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-03-31 19:22:59
# changed : 2020-11-17 16:09:18
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

    return Model(inputs=[hist_in, meta_in], outputs=x, name=name)
