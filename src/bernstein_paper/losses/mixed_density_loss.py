#!env python3
# -*- coding: utf-8 -*-
# AUTHOR INFORMATION ##########################################################
# file   : mixed_density_loss.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-03-31 19:22:59
# COPYRIGHT ###################################################################
# NEEDS TO BE DISCUSSED WHEN RELEASED!
#
# PROJECT DESCRIPTION #########################################################
#
# NOTE: this project is following the PEP8 style guide
#
# Bla bla...
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-05-17 14:55:44
#  changes made : using JointDistributionSequential
# modified by   : Marcel Arpogaus
# modified time : 2020-03-31 19:22:59
#  changes made : newly written
###############################################################################

# PYTHON AUTHORSHIP INFORMATION ###############################################
# ref.: https://stackoverflow.com/questions/1523427

"""baseline.py: [Description]"""

__author__ = ["Marcel Arpogaus"]
# __authors__ = ["author_one", "author_two" ]
# __contact__ = "kontakt@htwg-konstanz.de"

# __copyright__ = ""
# __license__ = ""

__date__ = "2020-03-31 19:22:59"
# __status__ = ""
# __version__ = ""

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow.keras.losses import Loss

from bernstein_paper.distributions import MixedNormal


class MixtedDensityLoss(Loss):
    def __init__(
            self,
            **kwargs):

        super().__init__(**kwargs)

    def call(self, y, pvector):

        dist = MixedNormal(pvector)
        y = tf.squeeze(y)
        nll = -dist.log_prob(y)

        return nll
