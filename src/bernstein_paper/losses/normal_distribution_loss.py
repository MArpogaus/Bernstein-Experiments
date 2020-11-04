import tensorflow as tf

from tensorflow.keras.losses import Loss

from bernstein_paper.distributions import NormalDistribution


class NormalDistributionLoss(Loss):
    def __init__(
            self,
            **kwargs):

        self.normal_distribution = NormalDistribution()

        super().__init__(**kwargs)

    def call(self, y, pvector):

        dist = self.normal_distribution(pvector)
        nll = -dist.log_prob(tf.squeeze(y))

        return nll
