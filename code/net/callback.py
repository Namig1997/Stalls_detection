import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


class ReduceLROnPlateauRestore(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(ReduceLROnPlateauRestore, self).__init__(*args, **kwargs)
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not self.monitor_op(current, self.best):
            if not self.in_cooldown():
                if self.wait+1 >= self.patience:
                    self.model.set_weights(self.best_weights)
        else:
            self.best_weights = self.model.get_weights()

        super().on_epoch_end(epoch, logs)
