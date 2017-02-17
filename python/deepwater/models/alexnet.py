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

        conv1 = conv11x11(x, 96, stride=4, padding='VALID')
        pool1 = max_pool_3x3(conv1, padding='VALID')
        lrn1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.0001, beta=0.75)

        conv2 = conv5x5(lrn1, 256)
        pool2 = max_pool_3x3(conv2, padding='VALID')
        lrn2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.0001, beta=0.75)

        conv3 = conv3x3(lrn2, 384)
        conv4 = conv3x3(conv3, 384)
        conv5 = conv3x3(conv4, 256)

        pool3 = max_pool_3x3(conv5)
        lrn3 = tf.nn.lrn(pool3, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        dims = lrn3.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        flatten = tf.reshape(lrn3, [-1, int(flatten_size)])

        fc1 = fc(flatten, [int(flatten_size), 4096])
        relu1 = tf.nn.relu(fc1)
        # dropout1 = tf.nn.dropout(relu1, 0.5)

        fc2 = fc(relu1, [4096, 4096])
        relu2 = tf.nn.relu(fc2)
        # dropout2 = tf.nn.dropout(relu2, 0.5)

        y = fc(relu2, [4096, classes])

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
