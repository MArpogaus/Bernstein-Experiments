import tensorflow as tf

from tensorflow.keras.losses import Loss

from tensorflow_probability import distributions as tfd


class NegativeLogarithmicLikelihood(Loss):
    def __init__(
            self,
            distribution_class,
            name='negative_logarithmic_likelihood',
            **kwargs):
        self.distribution_class = distribution_class
        super().__init__(name=name, **kwargs)

    def call(self, y, pvector):
        dist = tfd.Independent(self.distribution_class(pvector))
        nll = -dist.log_prob(tf.squeeze(y))
        return nll
