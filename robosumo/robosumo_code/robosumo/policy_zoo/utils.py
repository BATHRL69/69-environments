"""
A variety of utilities updated for TensorFlow 2.x.
"""

import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class RunningMeanStd:
    def __init__(self, epsilon=1e-2, shape=()):
        self.shape = shape

        self._sum = tf.Variable(
            initial_value=tf.zeros(shape, dtype=tf.float32),
            trainable=False,
            name="sum")
        self._sumsq = tf.Variable(
            initial_value=tf.zeros(shape, dtype=tf.float32),
            trainable=False,
            name="sumsq")
        self._count = tf.Variable(
            initial_value=epsilon,
            trainable=False,
            name="count",
            dtype=tf.float32)

    @property
    def mean(self):
        return self._sum / self._count

    @property
    def std(self):
        var_est = (self._sumsq / self._count) - tf.square(self.mean)
        return tf.sqrt(tf.maximum(var_est, 1e-2))

    def update(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        batch_sum = tf.reduce_sum(x, axis=0)
        batch_sumsq = tf.reduce_sum(tf.square(x), axis=0)
        batch_count = tf.cast(tf.shape(x)[0], tf.float32)

        self._sum.assign_add(batch_sum)
        self._sumsq.assign_add(batch_sumsq)
        self._count.assign_add(batch_count)

class DiagonalGaussian:
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        self.distribution = tfd.MultivariateNormalDiag(
            loc=self.mean,
            scale_diag=self.std)

    def sample(self):
        return self.distribution.sample()

    def mode(self):
        return self.mean

def dense(x, size, name, weight_init=None, bias=True):
    layer = tf.keras.layers.Dense(
        units=size,
        activation=None,
        use_bias=bias,
        kernel_initializer=weight_init,
        name=name)
    return layer(x)

def switch(condition, if_exp, else_exp):
    condition = tf.cast(condition, tf.bool)
    x = tf.cond(condition, lambda: if_exp, lambda: else_exp)
    return x

def load_params(path):
    return np.load(path, allow_pickle=True)

def set_from_flat(var_list, flat_params):
    shapes = [v.shape.as_list() for v in var_list]
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    assert flat_params.size == total_size, "Size mismatch"

    start = 0
    for shape, v in zip(shapes, var_list):
        size = int(np.prod(shape))
        param_values = flat_params[start:start + size].reshape(shape)
        v.assign(param_values)
        start += size
