import tensorflow as tf

def is_training():
    return tf.get_default_graph().get_tensor_by_name("global_is_training:0")

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

    initialization = tf.truncated_normal(shape, stddev=stddev)

    return tf.Variable(initialization, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    var = tf.Variable(initial, name=name)
    return var

def conv11x11(x, filters, **kwds):
    return conv(x, 11, 11, filters, **kwds)

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

def conv(x, w, h, filters, stride=1, padding="SAME", batch_norm = False, activation="relu", normalizer_fn = None, normalizer_params = None):
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = { 'is_training': is_training() }

    out = tf.contrib.layers.convolution2d(inputs=x, num_outputs=filters, kernel_size=[w, h],
                                          stride=stride, padding=padding,
                                          weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          biases_initializer=tf.contrib.layers.xavier_initializer(),
                                          #weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.34, mode='FAN_IN', uniform=False),
                                          #biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.34, mode='FAN_IN', uniform=False),
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          trainable=True)
    if activation == "relu":
        out = tf.nn.relu(out)
    elif activation == "tanh":
        out = tf.nn.tanh(out)
    return out


def max_pool_2x2(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def max_pool_3x3(x, stride=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)

def constant_initializer(value=0.01):
     return lambda shape, dtype, partition_info: tf.fill(shape, value)

def fc_bn(x, num_outputs):
    return fc(x, num_outputs, normalizer_fn = tf.contrib.layers.batch_norm, normalizer_params = { 'is_training': is_training() })

def fc(x, num_outputs, normalizer_fn = None, normalizer_params = None, activation_fn=None):
    out = tf.contrib.layers.fully_connected(inputs=x, num_outputs=num_outputs,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            #weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.34, mode='FAN_IN', uniform=False),
                                            #biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.34, mode='FAN_IN', uniform=False),
                                            biases_initializer=tf.contrib.layers.xavier_initializer(),
                                            activation_fn=activation_fn,
                                            normalizer_fn=normalizer_fn,
                                            normalizer_params=normalizer_params,
                                            trainable=True)
    return out