import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
from deepwater.models.nn import block, max_pool_2x2, get_fans, fc


# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, mean=0.1, stddev=0.1)
#     var = tf.Variable(initial, name=name)
#     return var
#
#
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape)
#     var = tf.Variable(initial, name=name)
#     return var
#
#
# def conv2d(x, weights):
#     return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')


class AlexNet(BaseImageClassificationModel):
    def __init__(self, width, height, channels, classes):
        super(AlexNet, self).__init__()

        self._number_of_classes = classes

        self._inputs = tf.placeholder(tf.float32, [None, width * height * channels])

        out = tf.reshape(self._inputs, [-1, width, height, channels])

        if width < 224:
            out = tf.image.resize_images(out, [48, 48])
        else:
            out = tf.image.resize_images(out, [224, 224])

        with tf.name_scope('conv1'):
            out = block(out, [11, 11, channels, 96])
            out = tf.nn.max_pool(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool1')

        with tf.name_scope('conv2'):
            out = block(out, [5, 5, 96, 256])
            out = tf.nn.max_pool(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool2')

        with tf.name_scope('conv3'):
            out = block(out, [3, 3, 256, 384])
        with tf.name_scope('conv4'):
            out = block(out, [3, 3, 384, 192])
        with tf.name_scope('conv5'):
            out = block(out, [3, 3, 192, 256])
            out = tf.nn.max_pool(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool5')

        dims = out.get_shape().as_list()
        prod = 1
        for d in dims[1:]:
            prod *= d

        out = tf.reshape(out, [-1, prod])

        out = fc(out, [prod, 4096])
        out = tf.nn.relu(out)

        out = fc(out, [4096, 4096])
        out = tf.nn.relu(out)

        out = fc(out, [4096, classes])

        self._logits = out

        self._predictions = tf.nn.softmax(self._logits)

    @property
    def train_dict(self):
        return {}

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
