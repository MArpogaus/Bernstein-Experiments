#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : test_quantile_regression_distribution_wrapper.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-22 10:46:18
# changed : 2020-12-03 14:15:04
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

from bernstein_paper.distributions import QuantileRegressionDistributionWrapper


class QuantileRegressionDistributionWrapperTest(tf.test.TestCase):

    def gen_dist(self, batch_shape):
        if batch_shape != []:
            n = tfd.Normal(loc=tf.zeros((batch_shape)),
                           scale=tf.ones((batch_shape)))
        else:
            n = tfd.Normal(loc=tf.zeros((1)), scale=tf.ones((1)))

        tol = 1e-7
        shape = [...] + [None] * len(batch_shape)
        p = tf.linspace(tol, 1 - tol, 100)[shape]
        q = n.quantile(p)
        perm = list(range(1, q.ndim)) + [0]
        pv = tf.transpose(q, perm)
        bs = QuantileRegressionDistributionWrapper(
            pv, constrain_quantiles=tf.identity)
        return n, bs

    def test_dist(self, batch_shape=[]):

        normal_dist, qr_dist = self.gen_dist(batch_shape)

        for input_shape in [[1], [1, 1], [1] + batch_shape]:

            # Check the distribution.
            self.assertEqual(normal_dist.batch_shape, qr_dist.batch_shape)
            self.assertEqual(normal_dist.event_shape, qr_dist.event_shape)
            self.assertAllClose(
                normal_dist.quantile(0.4 * tf.ones(input_shape)),
                qr_dist.quantile(0.4 * tf.ones(input_shape)),
                rtol=1e-4, atol=1e-5)
            self.assertAllClose(normal_dist.mean(), qr_dist.mean(),
                                rtol=1e-2, atol=1e-5)

            mu = normal_dist.mean()
            self.assertAllClose(normal_dist.prob(mu),
                                qr_dist.prob(mu),
                                rtol=1e-4, atol=1e-4)
            self.assertAllClose(normal_dist.log_prob(mu),
                                qr_dist.log_prob(mu),
                                rtol=1e-4, atol=1e-4)
            p_min = 0.1
            min_q = normal_dist.quantile(p_min)
            max_q = normal_dist.quantile(1 - p_min)
            q = tf.linspace(min_q, max_q, 100)

            self.assertAllClose(normal_dist.cdf(q),
                                qr_dist.cdf(q),
                                rtol=1e-4, atol=1e-4)
            self.assertAllClose(normal_dist.log_cdf(q),
                                qr_dist.log_cdf(q),
                                rtol=1e-3, atol=1e-3)

    def test_dist_batch(self):
        self.test_dist(batch_shape=[32])

    def test_dist_multi(self):
        self.test_dist(batch_shape=[32, 48])
