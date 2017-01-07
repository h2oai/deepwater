import tensorflow as tf

from deepwater.models import BaseImageClassificationModel


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0.1, stddev=0.1)
    var = tf.Variable(initial, name=name)
    return var


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial)
    return var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class AlexNet(BaseImageClassificationModel):

    def __init__(self, width, height, channels, classes):
        super(AlexNet, self).__init__()

        self._number_of_classes = classes

        size = width * height * channels

        self._inputs = tf.placeholder(tf.float32, [None, width * height * channels])

        images = tf.reshape(self._inputs, [-1, width, height, channels])

        w_stride =  max(width / 32.0, 1.0)
        h_stride =  max(height / 32.0, 1.0)

        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, channels, 96], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, w_stride, h_stride, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)

            lrn = tf.nn.local_response_normalization(conv1)

            pool1 = tf.nn.max_pool(lrn,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool1')
            print(conv1.get_shape())
            print(pool1.get_shape())

        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)

            lrn2 = tf.nn.local_response_normalization(conv2)

            pool2 = tf.nn.max_pool(lrn2,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool2')
            print(conv2.get_shape())
            print(pool2.get_shape())

        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            print(conv3.get_shape())

        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 192],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            print(conv4.get_shape())

        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 256],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)


            # pool5
            pool5 = tf.nn.max_pool(conv5,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool5')

            print(pool5.get_shape())
    
        dim =  int(((((width/ 2) - 2) / 2) -1) / 2)

        flatten = tf.reshape(pool5, [-1, dim * dim * 256])

        W_fc6 = weight_variable([dim * dim * 256, 4096], 'fc6/W')
        b_fc6 = bias_variable([4096], 'fc6/bias')
        fc6 = tf.matmul(flatten, W_fc6) + b_fc6

        W_fc7 = weight_variable([4096, 4096], 'fc7/W')
        b_fc7 = bias_variable([4096], 'fc7/bias')
        fc7 = tf.matmul(fc6, W_fc7) + b_fc7

        W_fc8 = weight_variable([4096, classes], 'fc8/W')
        b_fc8 = bias_variable([classes], 'fc8/bias')

        self._logits = tf.matmul(fc7, W_fc8) + b_fc8

        self._predictions = tf.nn.softmax(self._logits)

    @property
    def name(self):
        return "AlexNet"

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
