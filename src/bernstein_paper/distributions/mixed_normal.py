#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : mixed_normal.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-05-15 10:44:23
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
# NOTES ######################################################################
#
# This project is following the
# [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-07-09 11:16:19
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-05-15 10:44:23
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_probability import distributions as tfd


class MixedNormal():
    def __init__(self):
        pass

    def __call__(self, pvector):

        mixture = self.gen_mixture(pvector)

        return mixture

    def slice_parameter_vectors(self, pvector):
        """ Returns an unpacked list of paramter vectors.
        """
        num_dist = pvector.shape[1]
        sliced_pvectors = []
        for d in range(num_dist):
            sliced_pvector = [pvector[:, d, p] for p in range(3)]
            sliced_pvectors.append(sliced_pvector)
        return sliced_pvectors

    def gen_mixture(self, out):
        pvs = self.slice_parameter_vectors(out)
        mixtures = []

        for pv in pvs:
            logits, locs, log_scales = pv
            scales = tf.math.softmax(log_scales)
            mixtures.append(
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=logits),
                    components_distribution=tfd.Normal(
                        loc=locs,
                        scale=scales))
            )

        joint = tfd.JointDistributionSequential(
            mixtures, name='joint_mixtures')
        blkws = tfd.Blockwise(joint)
        return blkws
