#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : multivariate_bernstein_flow_loss.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-03-31 19:22:59
# changed : 2020-11-02 09:55:07
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/)
#
# LICENSE #####################################################################
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
from tensorflow.keras.losses import Loss

from bernstein_paper.distributions import MultivariateBernsteinFlow


class MultivariateBernsteinFlowLoss(Loss):
    """
    This Keras Loss function implements the negative logarithmic likelihood for
    a bijective transformation model using Bernstein polynomials.
    """

    def __init__(
            self,
            **kwargs: dict):
        """
        Constructs a new instance of the Keras Loss function.

        :param      M:       Order of the used Bernstein polynomial bijector.
        :type       M:       int
        :param      kwargs:  Additional keyword arguments passed to the supper
                             class
        :type       kwargs:  dictionary
        """
        super().__init__(**kwargs)

    def call(self,
             y: tf.Tensor,
             pvector: tf.Tensor) -> tf.Tensor:
        """
        Evaluates the negative logarithmic likelihood given a sample y.

        :param      y:        A sample.
        :type       y:        Tensor
        :param      pvector:  The parameter vector for the normalizing flow.
        :type       pvector:  Tensor

        :returns:   negative logarithmic likelihood
        :rtype:     Tensor
        """
        flow = MultivariateBernsteinFlow(pvector)

        nll = -flow.log_prob(tf.squeeze(y))

        return nll
