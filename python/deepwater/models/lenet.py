import tensorflow as tf
import math


from deepwater.models import BaseImageClassificationModel

def weight_variable(shape, name):
    # Delving deep into Rectifier
    n = shape[0] 
    factor=2.0 
    stddev=math.sqrt(1.3 * factor/n)

    initialization = tf.truncated_normal(
        shape, mean=0.0, stddev=stddev)

    var = tf.Variable(initialization, name=name)
    return var

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    var = tf.Variable(initial)
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class LeNet(BaseImageClassificationModel):

    def __init__(self, width, height, channels, classes):
        super(LeNet, self).__init__()

        self._number_of_classes = classes

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")

        self._inputs = x

        W_conv1 = weight_variable([5, 5, channels, 32], 'conv1/W')
        b_conv1 = bias_variable([32], 'conv1/bias')

        x_image = tf.reshape(x, [-1, width, height, channels], 
                name="input_reshape")

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64], 'conv2/W')
        b_conv2 = bias_variable([64], 'conv2/bias')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        dim = int(width / 4.0)

        W_fc1 = weight_variable([dim * dim * 64, 1024], 'conv3/W')
        b_fc1 = bias_variable([1024], 'conv3/bias')

        h_pool2_flat = tf.reshape(h_pool2, [-1, dim * dim * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self._dropout = keep_prob = tf.placeholder_with_default(1.0, [], name="dropout")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, classes], 'fc1/W')
        b_fc2 = bias_variable([classes], 'fc1/bias')

        self._logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self._predictions = tf.nn.softmax(self._logits)

    @property
    def train_dict(self):
        return {
            self._dropout: 1.0,
        }

    @property
    def name(self):
        return "LeNet"

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

