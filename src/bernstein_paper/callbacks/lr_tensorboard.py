from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


# found in: https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
