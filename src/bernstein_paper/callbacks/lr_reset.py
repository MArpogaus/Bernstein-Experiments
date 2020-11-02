from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class LRReset(Callback):
    def __init__(self, lr, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

    def on_train_begin(self, epoch):
        old_lr = float(K.eval(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, max(self.lr, old_lr))
