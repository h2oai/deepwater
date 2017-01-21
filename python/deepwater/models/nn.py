import math
import numpy as np
import tensorflow as tf


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return float(fan_in), float(fan_out)


def weight_variable(shape, name):
    # Delving deep into Rectifier
    # http://arxiv.org/pdf/1502.01852v1.pdf)
    # fan_in, _ = get_fans(shape)
    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)

    # assert stddev > 0.001, stddev

    factor = 2.0
    stddev = math.sqrt(1.3 * factor / fan_in)

    initialization = tf.truncated_normal(shape, 0.0, stddev)

    return tf.Variable(initialization, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    var = tf.Variable(initial, name=name)
    return var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def max_pool_3x3(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


def block(x, kernel_shape):
    kernel = weight_variable(kernel_shape, "kernel")
    b = bias_variable([kernel_shape[-1]], "bias")
    out = conv2d(x, kernel) + b
    return tf.nn.relu(out)


def fc(x, shape):
    W = weight_variable(shape, "weight")
    b = bias_variable([shape[-1]], "bias")
    return tf.matmul(x, W) + b
