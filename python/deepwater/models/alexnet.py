import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
from deepwater.models.nn import fc, conv, conv3x3, conv5x5, conv11x11, max_pool_3x3


class AlexNet(BaseImageClassificationModel):
    def __init__(self, width, height, channels, classes):
        super(AlexNet, self).__init__()

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        x = tf.reshape(self._inputs, [-1, width, height, channels])

        self._number_of_classes = classes

        if width < 224:
            x = tf.image.resize_images(x, [48, 48])
        else:
            x = tf.image.resize_images(x, [224, 224])

        out = conv11x11(x, 96, stride=4, padding='VALID')
        out = max_pool_3x3(out, padding='VALID')
        out = tf.nn.lrn(out, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        out = conv5x5(out, 256)
        out = max_pool_3x3(out, padding='VALID')
        out = tf.nn.lrn(out, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        out = conv3x3(out, 384)
        out = conv3x3(out, 192)
        out = conv3x3(out, 256)

        out = max_pool_3x3(out)
        out = tf.nn.lrn(out, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        out = tf.reshape(out, [-1, int(flatten_size)])

        # fully connected
        out = fc(out, [int(flatten_size), 4096])
        out = tf.nn.relu(out)
        out = fc(out, [4096, 4096])
        out = tf.nn.relu(out)
        y = fc(out, [4096, classes])

        self._logits = y

        if classes > 1:
            self._predictions = tf.nn.softmax(y)
        else:
            self._predictions = self._logits

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
