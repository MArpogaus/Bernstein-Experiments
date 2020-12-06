#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : mean_squared_error.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-11-24 16:01:15
# changed : 2020-12-06 13:05:08
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
import tensorflow as tf

from tensorflow_probability import distributions as tfd

class MeanSquaredError(tf.keras.metrics.MeanSquaredError):

    def __init__(self,
                 distribution_class,
                 independent = True,
                 name='mean_squared_error',
                 scale=1.,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.distribution_class = distribution_class
        self.scale = scale
        self.independent = independent

    def update_state(self, y_true, pvector, sample_weight=None):
        dist = self.distribution_class(pvector)
        if self.independent:
            dist = tfd.Independent(dist)

        mean = dist.mean()
        super().update_state(
            y_true * self.scale,
            mean * self.scale,
            sample_weight
        )
