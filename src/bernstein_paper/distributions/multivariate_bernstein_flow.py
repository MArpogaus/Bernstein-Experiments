#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : multivariate_bernstein_flow.py
# brief   : [Description]
#
# author  : marcel
# created : 2020-10-30 20:13:08
# changed : 2020-11-19 15:49:20
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

from bernstein_flow.distributions import BernsteinFlow


class MultivariateBernsteinFlow(tfd.Blockwise):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(self,
                 pvector: tf.Tensor,
                 distribution: tfd.Distribution = tfd.Normal(loc=0., scale=1.)
                 ) -> tfd.Distribution:
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      pvector:       The paramter vector.
        :type       pvector:       Tensor
        :param      distribution:  The base distribution to use.
        :type       distribution:  Distribution

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        num_dist = pvector.shape[1]

        flows = []
        for d in range(num_dist):
            flow = BernsteinFlow(pvector[:, d])
            flows.append(flow)

        joint = tfd.JointDistributionSequential(flows, name='joint_bs_flows')
        super().__init__(flows, name='MultivariateBernsteinFlow')

    def _stddev(self):
        return self._flatten_and_concat_event(self._distribution.stddev())

    def _quantile(self, value):
        qs = [d._quantile(value) for d in self.distributions]
        return self._flatten_and_concat_event(qs)

    def _cdf(self, value):
        qs = [d._cdf(value) for d in self.distributions]
        return self._flatten_and_concat_event(qs)
