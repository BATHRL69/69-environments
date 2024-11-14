import copy
import numpy as np
import tensorflow as tf


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            self._sum = tf.compat.v1.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.compat.v1.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.compat.v1.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.cast(self._sum / self._count, tf.float32)
            var_est = tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))

    def mode(self):
        return self.mean


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.compat.v1.get_variable(name + "/w", [x.get_shape()[1], size],
                                  initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.compat.v1.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, tf.bool),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


def load_params(path):
    return np.load(path)


def set_from_flat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.compat.v1.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assign = tf.compat.v1.assign(v, tf.reshape(theta[start:start + size], shape))
        assigns.append(assign)
        start += size
    op = tf.group(*assigns)
    tf.compat.v1.get_default_session().run(op, {theta: flat_params})
