import tensorflow as tf
import tensorflow_probability as tfp


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, mse_scale=1, pearson_scale=1, likelihood_scale=1):
        self.mse_scale = mse_scale
        self.pearson_scale = pearson_scale
        self.likelihood_scale = likelihood_scale
        super().__init__()

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred.mean() - y_true)) * self.mse_scale
        neg_log_likelihood = self.negative_loglikelihood(y_true, y_pred) * self.likelihood_scale
        neg_scaled_pearson = -tfp.stats.correlation(y_true, y_pred) * self.pearson_scale
        return neg_log_likelihood + mse + neg_scaled_pearson

    def negative_loglikelihood(self, targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)
    