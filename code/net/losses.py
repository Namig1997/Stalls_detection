import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weight=10., 
            name="weighted_binary_crossentropy", **kwargs):
        super(WeightedBinaryCrossentropy, self).__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        return -tf.math.reduce_mean(
            self.weight * y_true * tf.math.log(y_pred + 1e-9) + \
            (1. - y_true) * tf.math.log(1. - y_pred + 1e-9))