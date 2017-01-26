import math
import tensorflow as tf


# from: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def batch_norm(x, n_out, scope=''):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """

    from tensorflow.contrib.layers import batch_norm as layers_batch_norm
    phase_train = is_training()
    batch_norm_decay=0.997
    batch_norm_epsilon = 1e-5
    return layers_batch_norm(x, epsilon=batch_norm_epsilon)


    # with tf.variable_scope(scope):
    #     batch_norm_decay=0.997
    #
    #     beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
    #                        name='beta', trainable=True)
    #     gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
    #                         name='gamma', trainable=True)
    #     batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    #     ema = tf.train.ExponentialMovingAverage(decay=batch_norm_decay)
    #
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)
    #
    #     with tf.control_dependencies([phase_train]):
    #
    #         mean, var = tf.cond(tf.logical_and(True, phase_train),
    #                             mean_var_with_update,
    #                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #
    #     batch_norm_epsilon = 1e-5
    #     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, batch_norm_epsilon)
    # return normed
    #



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


def conv1x1(x, filters, **kwds):
    return conv(x, 1, 1, filters, **kwds)


def conv5x5(x, filters, **kwds):
    return conv(x, 5, 5, filters, **kwds)


def conv3x3(x, filters, **kwds):
    return conv(x, 3, 3, filters, **kwds)


def conv1x3(x, filters, **kwds):
    return conv(x, 1, 3, filters, **kwds)


def conv3x1(x, filters, **kwds):
    return conv(x, 3, 1, filters, **kwds)


def conv1x7(x, filters, **kwds):
    return conv(x, 1, 7, filters, **kwds)


def conv7x1(x, filters, **kwds):
    return conv(x, 7, 1, filters, **kwds)


def conv(x, w, h, filters, stride=1, padding="SAME"):
    channels = x.get_shape().as_list()[3]

    kernel_shape = [w, h, channels, filters]
    kernel = weight_variable(kernel_shape, "kernel")

    # you don't need to add bias if you are using BatchNormalization
    # b = bias_variable([filters], "bias")
    x = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=padding)

    x = batch_norm(x, filters)

    return tf.nn.relu(x)


def max_pool_2x2(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def max_pool_3x3(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


# def xblock(x, kernel_shape):
#     kernel = weight_variable(kernel_shape, "kernel")
#     b = bias_variable([kernel_shape[-1]], "bias")
#     out = conv2d(x, kernel) + b
#     return tf.nn.relu(out)


def fc(x, shape):
    W = weight_variable(shape, "weight")
    #b = bias_variable([shape[-1]], "bias")
    x = tf.matmul(x, W)
    return batch_norm(x, shape[-1])


def is_training():
    return tf.get_default_graph().get_tensor_by_name("global_is_training:0")
    # with tf.variable_scope('') as scope:
    #     scope.reuse_variables()
    #     return tf.get_variable("is_training",
    #                            initializer=lambda *args, **kwds: False, shape=[],
    #                            trainable=False,
    #                            dtype=tf.bool)
