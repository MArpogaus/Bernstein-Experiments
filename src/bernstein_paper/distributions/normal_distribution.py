# -*- coding: utf-8 -*-
#!/usr/bin/python
# Python Version 3.8.3

import tensorflow as tf

from tensorflow_probability import distributions as tfd


class NormalDistribution():
    def __init__(self):
        pass

    def __call__(self, pvector):

        mixture = self.gen_dist(pvector)

        return mixture

    def slice_parameter_vectors(self, pvector):
        num_dist = pvector.shape[1]
        sliced_pvectors = []
        for d in range(num_dist):
            sliced_pvector = [pvector[:, d, p] for p in range(2)]
            sliced_pvectors.append(sliced_pvector)
        return sliced_pvectors

    def gen_dist(self, out):
        pvs = self.slice_parameter_vectors(out)
        normales = []

        for pv in pvs:
            locs, log_scales = pv
            scales = 1e-3 + tf.math.softplus(0.05 * log_scales)
            normales.append(
                tfd.Normal(
                    loc=locs,
                    scale=scales
                ))

        joint = tfd.JointDistributionSequential(
            normales, name='joint_normales')
        blkws = tfd.Blockwise(joint)
        return blkws
