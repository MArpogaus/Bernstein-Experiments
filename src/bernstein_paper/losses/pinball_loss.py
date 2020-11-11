import tensorflow as tf

from tensorflow.keras.losses import Loss


class PinballLoss(Loss):
    def __init__(
            self,
            **kwargs):

        super().__init__(**kwargs)

    def slice_parameter_vectors(self, pvector):
        num_qunantiles = pvector.shape[1]
        sliced_pvectors = []
        for d in range(num_qunantiles):
            sliced_pvector = [pvector[:, d, p] for p in range(2)]
            sliced_pvectors.append(sliced_pvector)
        return sliced_pvectors

    def pinball_loss(y_true, y_pred, q):
        ''' Pinball loss for Tensorflow Backend '''
        error = tf.subtract(y_true, y_pred)
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

    def call(self, y, pvector):
        pvs = self.slice_parameter_vectors(pvector)

        losses = []
        for i, pv in enumerate(pvs):
            q = i / len(pvs)
            losses.append(self.pinball_loss(y, pv, q))

        return tf.reduce_mean(losses)
