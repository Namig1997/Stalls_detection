import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf



# threshold_mcc=0.5
# def mcc(y_true, y_pred):
#   predicted = tf.cast(tf.greater(y_pred, threshold_mcc), tf.float32)
#   true_pos = tf.math.count_nonzero(predicted * y_true)
#   true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
#   false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
#   false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
#   x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
#       * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
#   return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / \
#       tf.maximum(tf.sqrt(x), 1e-9)


class MCC(tf.keras.metrics.Metric):
    """Mattheus correlation coefficient"""
    def __init__(self, threshold=0.5, name="mcc", **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.TP = tf.keras.metrics.TruePositives()
        self.TN = tf.keras.metrics.TrueNegatives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.FN = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(tf.greater(y_pred, self.threshold), tf.float32)
        self.TP.update_state(y_true, y_pred_bin, sample_weight=sample_weight)
        self.TN.update_state(y_true, y_pred_bin, sample_weight=sample_weight)
        self.FP.update_state(y_true, y_pred_bin, sample_weight=sample_weight)
        self.FN.update_state(y_true, y_pred_bin, sample_weight=sample_weight)

    def result(self):
        TP = self.TP.result()
        TN = self.TN.result()
        FP = self.FP.result()
        FN = self.FN.result()
        return tf.cast((TP*TN)-(FP*FN), tf.float32) / \
            tf.maximum(tf.sqrt(tf.cast((TP+FP)*(TP+FP)*(TN+FP)*(TN+FN), tf.float32)), 1e-9)

    def reset_states(self):
        self.TP.reset_states()
        self.TN.reset_states()
        self.FP.reset_states()
        self.FN.reset_states() 