#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : quantile_regression_distribution_wrapper.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-11-26 14:21:13
# changed : 2020-11-26 14:50:54
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

import scipy.interpolate as I

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization

from ..losses import PinballLoss

class QuantileRegressionDistributionWrapper(tfd.Distribution):

    def __init__(self,
                 quantiles,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='QuantileDistributionWrapper'):

        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype(
                [quantiles], dtype_hint=tf.float32)

            self.quantiles = tensor_util.convert_nonref_to_tensor(
                quantiles, dtype=dtype, name='quantiles')

            assert self.quantiles.shape[-1] == 100, '100 Qunatiles reqired'

            self.quantiles = PinballLoss.constrain_quantiles(self.quantiles)

            self._pdf_sp, self._cdf_sp = self.make_interp_spline()

            super().__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name)

    def make_interp_spline(self):
        """
        Generates the Spline Interpolation.
        """
        percentiles = np.linspace(0., 1., 100, dtype=np.float32)
        quantiles = self.quantiles.numpy().copy()

        # float_min = np.finfo(np.float32).min * np.ones_like(quantiles[...,:1])
        # float_max = np.finfo(np.float32).max * np.ones_like(quantiles[...,-1:])

        #quantiles[...,0] = quantiles[...,1] - 5 * np.diff(quantiles)[...,1]
        #quantiles[...,-1] = quantiles[...,-2] + 5 * np.diff(quantiles)[...,-2]

        #x = np.concatenate([float_min, quantiles, float_max],axis=-1)
        #y = np.concatenate([percentiles[...,:1], percentiles, percentiles[...,-1:]],axis=-1)

        x = quantiles
        y = percentiles

        x = x.reshape(-1, x.shape[-1])

        x_min = np.min(x, axis=-1)  # [shape]
        x_max = np.max(x, axis=-1)  # [shape]

        cdf_sp = [I.make_interp_spline(
            y=np.squeeze(y),
            x=np.squeeze(x[i]),
            k=3,
            bc_type=([(1, 0.0)], [(1, 0.0)]),
            # assume_sorted=True
        ) for i in range(x.shape[0])]
        pdf_sp = [s.derivative(1) for s in cdf_sp]

        def pdf_sp_fn(x):
            y = []
            z_clip = np.clip(x, x_min, x_max)
            for i, ip in enumerate(pdf_sp):
                y.append(ip(z_clip[..., i]).astype(np.float32))
            y = np.stack(y, axis=-1)
            return y

        def cdf_sp_fn(x):
            y = []
            z_clip = np.clip(x, x_min, x_max)
            for i, ip in enumerate(cdf_sp):
                y.append(ip(z_clip[..., i]).astype(np.float32))
            y = np.stack(y, axis=-1)
            return y

        return pdf_sp_fn, cdf_sp_fn

    def reshape_out(self, sample_shape, y):
        output_shape = prefer_static.broadcast_shape(
            sample_shape, self.batch_shape)
        return tf.reshape(y, output_shape)

    def _eval_spline(self, x, attr):
        batch_rank = tensorshape_util.rank(self.batch_shape)
        sample_shape = x.shape

        if x.shape[-batch_rank:] == self.batch_shape:
            shape = list(x.shape[:-batch_rank]) + [-1]
            x = tf.reshape(x, shape)
        else:
            x = x[..., None]

        return self.reshape_out(sample_shape, getattr(self, attr)(x))

    def _batch_shape(self):
        shape = tf.TensorShape(prefer_static.shape(self.quantiles)[:-1])
        return tf.broadcast_static_shape(shape, tf.TensorShape([1]))

    def _event_shape(self):
        return tf.TensorShape([])

    def _log_prob(self, x):
        return np.log(self.prob(x))

    def _prob(self, x):
        return self._eval_spline(x, '_pdf_sp')

    def _log_cdf(self, x):
        return np.log(self.cdf(x))

    def _cdf(self, x):
        return self._eval_spline(x, '_cdf_sp')

    def _mean(self):
        return self.quantiles[..., 50]

    def _quantile(self, p):
        input_shape = p.shape
        q = self.quantiles
        perm = tf.concat([[q.ndim - 1], tf.range(0, q.ndim - 1)], 0)
        q = tfp.math.interp_regular_1d_grid(
            p,
            x_ref_min=0.,
            x_ref_max=1.,
            y_ref=tf.transpose(q, perm),
            axis=0)

        return self.reshape_out(input_shape, q)
