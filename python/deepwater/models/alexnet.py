import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
from deepwater.models.nn import fc, conv, conv3x3


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
            out = conv(out, 11, 11, 96)
            out = tf.nn.max_pool(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool1')

        with tf.name_scope('conv2'):
            out = conv(out, 5, 5, 256)
            out = tf.nn.max_pool(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool2')

        with tf.name_scope('conv3'):
            out = conv3x3(out, 384)
        with tf.name_scope('conv4'):
            out = conv3x3(out, 192)
        with tf.name_scope('conv5'):
            out = conv3x3(out, 256)
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
