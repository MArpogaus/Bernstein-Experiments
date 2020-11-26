#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : mixed_normal.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-11-23 17:30:45
# changed : 2020-11-25 13:40:19
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


class MixedNormal(tfd.MixtureSameFamily):

    def __init__(self, pvector):

        logits = pvector[..., 0]
        locs = pvector[..., 1]
        scales = tf.math.softplus(pvector[..., 2])

        super().__init__(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.Normal(
                loc=locs,
                scale=scales),
            name='MixedNormal')
