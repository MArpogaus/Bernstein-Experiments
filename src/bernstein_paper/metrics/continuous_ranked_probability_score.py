#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : continuous_ranked_probability_score.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-11-24 16:03:02
# changed : 2020-12-03 09:27:30
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

from absl import logging


@tf.function
def trapez(y, x):
    d = x[1:] - x[:-1]
    return tf.reduce_sum(d * (y[1:] + y[:-1]) / 2.0, axis=0)


class ContinuousRankedProbabilityScore(tf.keras.metrics.Mean):
    def __init__(
        self,
        distribution_class,
        name="continuous_ranked_probability_score",
        scale=1.0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.distribution_class = distribution_class
        self.scale = scale
        self.tol = 1e-4

    def update_state(self, y_true, pvector, sample_weight=None):
        y_true = tf.squeeze(y_true)

        n_points = 10000

        dist = self.distribution_class(pvector)
        cdf = dist.cdf

        # Note that infinite values for xmin and xmax are valid, but
        # it slows down the resulting quadrature significantly.
        try:
            x_min = dist.quantile(self.tol)
            x_max = dist.quantile(1 - self.tol)
        except:
            x_min = -(10 ** (2 + y_true // 10))
            x_max = 10 ** (2 + y_true // 10)

        # make sure the bounds haven't clipped the cdf.
        warning = "CDF does not meet tolerance requirements at {} extreme(s)!"

        if tf.math.reduce_any(cdf(x_min) >= self.tol):
            logging.warning(warning.format("lower"))
        if tf.math.reduce_any(cdf(x_max) < (1.0 - self.tol)):
            logging.warning(warning.format("upper"))

            # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
            #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy

        def lhs(x):
            # left hand side of CRPS integral
            return tf.square(cdf(x))

        def rhs(x):
            # right hand side of CRPS integral
            return tf.square(1.0 - cdf(x))

        lhs_x = tf.linspace(x_min, y_true, n_points)
        lhs_int = trapez(lhs(lhs_x), lhs_x)

        rhs_x = tf.linspace(y_true, x_max, n_points)
        rhs_int = trapez(rhs(rhs_x), rhs_x)

        score = lhs_int + rhs_int

        return super().update_state(score, sample_weight=sample_weight)
