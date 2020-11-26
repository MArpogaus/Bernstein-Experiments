#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : normal_distribution.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-11-19 16:05:06
# changed : 2020-11-19 16:05:51
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


class NormalDistribution(tfd.Normal):

    def __init__(self, pvector):

        super().__init__(
            loc=pvector[..., 0],
            scale=1e-3 + tf.math.softplus(0.05 * pvector[..., 1]),
            name='NormalDistribution')
