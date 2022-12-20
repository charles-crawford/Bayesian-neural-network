import tensorflow as tf
import tensorflow_probability as tfp


class PearsonMetric(tf.keras.metrics.Metric):
    def __init__(self, name='pearson', **kwargs):
        super().__init__(name=name, **kwargs)
        self.y_true = self.add_weight(name='y_true', initializer='zeros')
        self.y_pred = self.add_weight(name='y_pred', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true = tf.cast(y_true, tf.float64)
        self.y_pred = tf.cast(y_pred, tf.float64)

    def result(self):
        # when monitoring for best model to write, the metric is minimized
        return -tfp.stats.correlation(self.y_true, self.y_pred)[0][0]
