import math
import numpy as np
import tensorflow as tf


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def weight_variable(shape, name):
    # Delving deep into Rectifier
    fan_in, fan_out = get_fans(shape)
    stddev = math.sqrt(2.0/ fan_in)

    initialization = tf.truncated_normal(
        shape, mean=0.0, stddev=stddev)

    return tf.Variable(initialization, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    var = tf.Variable(initial, name=name)
    return var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def block(x, kernel_shape):
    kernel = weight_variable(kernel_shape, "kernel")
    b = bias_variable([kernel_shape[-1]], "bias")
    out = conv2d(x, kernel) + b
    return tf.nn.relu(out)


def fc(x, shape):
    W = weight_variable(shape, "weight")
    b = bias_variable([shape[-1]], "bias")
    return tf.matmul(x, W) + b

