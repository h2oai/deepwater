import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm

from deepwater.models import BaseImageClassificationModel

from collections import namedtuple
from deepwater.models.nn import max_pool_3x3, fc, is_training, conv


class ResNet(BaseImageClassificationModel):
    def __init__(self, width=299, height=299, channels=3, classes=10):
        super(ResNet, self).__init__()

        assert width == height, "width and height must be the same"

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        x = tf.reshape(x, [-1, width, height, channels])

        self._number_of_classes = classes

        max_w = 299
        min_w = 299 // 3

        if width < min_w:
            x = tf.image.resize_images(x, [min_w, min_w])
        elif width == 299:
            pass # do nothing
        else:
            x = tf.image.resize_images(x, [max_w, max_w])

        # x = tf.reshape(x, [-1, width, height, channels],
        #              name="input_reshape")

        # normalizer_params = { 'is_training': is_training() }

        # adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/resnet.py

        # Configurations for each bottleneck group.

        # activation = tf.nn.relu
        BottleneckGroup = namedtuple('BottleneckGroup',
                                     ['num_blocks', 'num_filters', 'bottleneck_size'])
        groups = [
            BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64),
            # BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256)
        ]

        # First convolution expands to 64 channels
        with tf.variable_scope('conv_layer1'):
            # net = convolution2d(
            #     x, 64, 7, normalizer_fn=batch_norm, normalizer_params=normalizer_params, activation_fn=activation)
            net = conv(x, 7, 7, 64)

        # Max pool
        net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First chain of resnets
        with tf.variable_scope('conv_layer2'):
            # net = convolution2d(net, groups[0].num_filters, 1, padding='VALID')
            net = conv(net, 1, 1, groups[0].num_filters, padding='VALID')

        # Create the bottleneck groups, each of which contains `num_blocks`
        # bottleneck groups.
        for group_i, group in enumerate(groups):
            for block_i in range(group.num_blocks):
                name = 'group_%d/block_%d' % (group_i, block_i)

                # 1x1 convolution responsible for reducing dimension
                with tf.variable_scope(name + '/conv_in'):
                    # conv = convolution2d(
                    #     net,
                    #     group.bottleneck_size,
                    #     1,
                    #     padding='VALID',
                    #     activation_fn=activation,
                    #     normalizer_fn=batch_norm,
                    #     normalizer_params=normalizer_params)
                    convLayer = conv(net, 1, 1, group.bottleneck_size, padding='VALID')

                with tf.variable_scope(name + '/conv_bottleneck'):
                    # conv = convolution2d(
                    #     conv,
                    #     group.bottleneck_size,
                    #     3,
                    #     padding='SAME',
                    #     activation_fn=activation,
                    #     normalizer_fn=batch_norm,
                    #     normalizer_params=normalizer_params)
                    convLayer = conv(convLayer, 3, 3, group.bottleneck_size)

                # 1x1 convolution responsible for restoring dimension
                with tf.variable_scope(name + '/conv_out'):
                    input_dim = net.get_shape()[-1].value
                    # conv = convolution2d(
                    #     conv,
                    #     input_dim,
                    #     1,
                    #     padding='VALID',
                    #     activation_fn=activation,
                    #     normalizer_fn=batch_norm,
                    #     normalizer_params=normalizer_params)
                    convLayer = conv(convLayer, 1, 1, input_dim, padding='VALID')

                # shortcut connections that turn the network into its counterpart
                # residual function (identity shortcut)
                net = convLayer + net

            try:
                # upscale to the next group size
                next_group = groups[group_i + 1]
                with tf.variable_scope('block_%d/conv_upscale' % group_i):
                    # net = convolution2d(
                    #     net,
                    #     next_group.num_filters,
                    #     1,
                    #     activation_fn=None,
                    #     biases_initializer=None,
                    #     padding='SAME')
                    net = conv(net, 1, 1, next_group.num_filters)
            except IndexError:
                pass

        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(
            net,
            ksize=[1, net_shape[1], net_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        out = net

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        out = tf.reshape(out, [-1, int(flatten_size)])

        out = fc(out, classes)

        self._logits = out

        if classes > 1:
            self._predictions = tf.nn.softmax(self._logits)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {}

    @property
    def name(self):
        return "ResNet"

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @property
    def inputs(self):
        return self._inputs

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
