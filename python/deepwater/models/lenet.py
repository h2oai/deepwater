import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
# from deepwater.models.nn import weight_variable, bias_variable, max_pool_2x2, conv, fc
from deepwater.models.nn import  max_pool_2x2, conv, fc


class LeNet(BaseImageClassificationModel):

    def __init__(self, width, height, channels, classes):
        super(LeNet, self).__init__()

        self._number_of_classes = classes

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")

        self._inputs = x

        x_image = tf.reshape(x, [-1, width, height, channels],
                             name="input_reshape")

        with tf.variable_scope("conv1"):
            out = conv(x_image, 5, 5, 20, activation="tanh")
            out = max_pool_2x2(out)

        with tf.variable_scope("conv2"):
            out = conv(out, 5, 5, 50, activation="tanh")
            out = max_pool_2x2(out)

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        flatten = tf.reshape(out, [-1, int(flatten_size)])

        out = fc(flatten, [int(flatten_size), 500])
        out = tf.nn.tanh(out)

        self._logits = fc(out, [500, classes])

        self._predictions = tf.nn.softmax(self._logits)

    @property
    def train_dict(self):
        return {}

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
