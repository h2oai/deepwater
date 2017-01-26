import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
from deepwater.models.nn import weight_variable, bias_variable, max_pool_3x3, conv, fc


class LeNet(BaseImageClassificationModel):

    def __init__(self, width, height, channels, classes):
        super(LeNet, self).__init__()

        self._number_of_classes = classes

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")

        self._inputs = x

        x_image = tf.reshape(x, [-1, width, height, channels],
                             name="input_reshape")

        out = conv(x_image, 5, 5, 32)
        out = max_pool_3x3(out)

        out = conv(out, 5, 5, 64)
        out = max_pool_3x3(out)

        dim = int(width / 4.0)

        out = tf.reshape(out, [-1, dim * dim * 64])

        out = fc(out, [dim * dim * 64, 1024])
        out = tf.nn.relu(out)

        self._dropout = keep_prob = tf.placeholder_with_default(1.0, [], name="dropout")
        out = tf.nn.dropout(out, keep_prob)

        self._logits = fc(out, [1024, classes])

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
