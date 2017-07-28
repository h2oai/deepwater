import tensorflow as tf

from deepwater.models import BaseImageClassificationModel

from deepwater.models.utils import concat
from deepwater.models.nn import max_pool_3x3, fc
from deepwater.models.nn import conv5x5, conv3x3, conv1x1, conv1x7, conv7x1, conv1x3, conv3x1, conv

use_batch_norm = True


# ================================= INCEPTION V3 =================================

def inception7A(out, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2, num_5x5_red, num_5x5, proj):
    out1 = conv1x1(out, num_1x1, batch_norm=use_batch_norm)

    out2 = conv1x1(out, num_5x5_red, batch_norm=use_batch_norm)
    out2 = conv5x5(out2, num_5x5, batch_norm=use_batch_norm)

    out3 = conv1x1(out, num_3x3_red, batch_norm=use_batch_norm)
    out3 = conv3x3(out3, num_3x3_1, batch_norm=use_batch_norm)
    out3 = conv3x3(out3, num_3x3_2, batch_norm=use_batch_norm)

    out4 = avg_pool_3x3(out)
    out4 = conv1x1(out4, proj, batch_norm=use_batch_norm)

    return concat(3, [out1, out2, out3, out4])


def inception7B(out, num_3x3, num_d3x3_red, num_d3x3_1, num_d3x3_2):
    out1 = conv3x3(out, num_3x3, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out2 = conv1x1(out, num_d3x3_red, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, num_d3x3_1, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, num_d3x3_2, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out3 = max_pool_3x3(out, stride=2, padding="VALID")

    return concat(3, [out1, out2, out3])


def inception7C(out, num_1x1, num_d7_red, num_d7_1, num_d7_2, num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4, proj):
    out1 = conv1x1(out, num_1x1, batch_norm=use_batch_norm)

    out2 = conv1x1(out, num_d7_red, batch_norm=use_batch_norm)
    out2 = conv1x7(out2, num_d7_1, batch_norm=use_batch_norm)
    out2 = conv7x1(out2, num_d7_2, batch_norm=use_batch_norm)

    out3 = conv1x1(out, num_q7_red, batch_norm=use_batch_norm)
    out3 = conv7x1(out3, num_q7_1, batch_norm=use_batch_norm)
    out3 = conv1x7(out3, num_q7_2, batch_norm=use_batch_norm)
    out3 = conv7x1(out3, num_q7_3, batch_norm=use_batch_norm)
    out3 = conv1x7(out3, num_q7_4, batch_norm=use_batch_norm)

    out4 = avg_pool_3x3(out)
    out4 = conv1x1(out4, proj, batch_norm=use_batch_norm)

    return concat(3, [out1, out2, out3, out4])


def inception7D(out, num_3x3_red, num_3x3, num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3):
    out1 = conv1x1(out, num_3x3_red, batch_norm=use_batch_norm)
    out1 = conv3x3(out1, num_3x3, stride=2, batch_norm=use_batch_norm)

    out2 = conv1x1(out, num_d7_3x3_red, batch_norm=use_batch_norm)
    out2 = conv1x7(out2, num_d7_1, batch_norm=use_batch_norm)
    out2 = conv7x1(out2, num_d7_2, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, num_d7_3x3, stride=2, batch_norm=use_batch_norm)

    out3 = max_pool_3x3(out, stride=2)

    return concat(3, [out1, out2, out3])


def inception7E(out, num_1x1, num_d3_red, num_d3_1, num_d3_2, num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2, proj, pool_3x3):
    out1 = conv1x1(out, num_1x1, batch_norm=use_batch_norm)

    out2 = conv1x1(out, num_d3_red, batch_norm=use_batch_norm)
    out2_1 = conv1x3(out2, num_d3_1, batch_norm=use_batch_norm)
    out2_2 = conv3x1(out2, num_d3_2, batch_norm=use_batch_norm)

    out3 = conv1x1(out, num_3x3_d3_red, batch_norm=use_batch_norm)
    out3 = conv3x3(out3, num_3x3, batch_norm=use_batch_norm)
    out3_1 = conv1x3(out3, num_3x3_d3_1, batch_norm=use_batch_norm)
    out3_2 = conv3x3(out3, num_3x3_d3_2, batch_norm=use_batch_norm)

    out4 = pool_3x3(out, stride=1)
    out4 = conv1x1(out4, proj, batch_norm=use_batch_norm)

    return concat(3, [out1, out2_1, out2_2, out3_1, out3_2, out4])


class InceptionV3(BaseImageClassificationModel):
    def __init__(self, width=299, height=299, channels=3, classes=10):
        super(InceptionV3, self).__init__()

        self._number_of_classes = classes

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        with tf.variable_scope("reshape1"):
            x = tf.reshape(x, [-1, width, height, channels])

        max_w = 299
        min_w = 299 // 3

        with tf.variable_scope("resize1"):
            if width < min_w:
                x = tf.image.resize_images(x, [min_w, min_w])
            elif width > 299:
                x = tf.image.resize_images(x, [max_w, max_w])

        # Stage 1
        out = conv3x3(x, 32, stride=2, padding="VALID", batch_norm=use_batch_norm)
        out = conv3x3(out, 32, padding="VALID", batch_norm=use_batch_norm)
        out = conv3x3(out, 64, batch_norm=use_batch_norm)
        out = max_pool_3x3(out, padding="VALID")

        # Stage 2
        out = conv3x3(out, 80, padding="VALID", batch_norm=use_batch_norm)
        out = conv3x3(out, 192, padding="VALID", batch_norm=use_batch_norm)
        out = max_pool_3x3(out, padding="VALID")

        # Stage 3
        out = inception7A(out, 64, 64, 96, 96, 48, 64, 32)
        out = inception7A(out, 64, 64, 96, 96, 48, 64, 64)
        out = inception7A(out, 64, 64, 96, 96, 48, 64, 64)
        out = inception7B(out, 384, 64, 96, 96)

        # Stage 4
        out = inception7C(out, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192)
        out = inception7C(out, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        out = inception7C(out, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        out = inception7C(out, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192)
        out = inception7D(out, 192, 320, 192, 192, 192, 192)

        # Stage 5
        out = inception7E(out, 320, 384, 384, 384, 448, 384, 384, 384, 192, avg_pool_3x3)
        out = inception7E(out, 320, 384, 384, 384, 448, 384, 384, 384, 192, max_pool_3x3)

        out = tf.nn.avg_pool(out, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME')

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        with tf.variable_scope("reshape2"):
            out = tf.reshape(out, [-1, int(flatten_size)])

        with tf.variable_scope("fc1"):
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
        return "inception_bn"

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


# ================================= INCEPTION V4 =================================

def stem(x):
    """
    Stem fo the pure InceptionV4 and Inception-ResNet-V2
    """
    out = conv3x3(x, 32, stride=2, padding="VALID", batch_norm=use_batch_norm)
    out = conv3x3(out, 32, stride=1, padding="VALID", batch_norm=use_batch_norm)
    out = conv3x3(out, 64, stride=1, padding="SAME", batch_norm=use_batch_norm)

    out1 = max_pool_3x3(out, stride=2, padding="VALID")
    out2 = conv3x3(out, 96, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out = concat(3, [out1, out2])

    out1 = conv1x1(out, 64, batch_norm=use_batch_norm)
    out1 = conv3x3(out1, 96, padding="VALID", batch_norm=use_batch_norm)

    out2 = conv1x1(out, 64, batch_norm=use_batch_norm)
    out2 = conv1x7(out2, 64, batch_norm=use_batch_norm)
    out2 = conv7x1(out2, 64, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, 96, padding="VALID", batch_norm=use_batch_norm)

    out = concat(3, [out1, out2])

    out1 = conv3x3(out, 192, stride=2, padding="VALID", batch_norm=use_batch_norm)
    out2 = max_pool_3x3(out, stride=2, padding="VALID")

    out = concat(3, [out1, out2])

    return out


def inceptionA(out):
    out1 = conv1x1(out, 96, batch_norm=use_batch_norm)

    out2 = conv1x1(out, 64, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, 96, batch_norm=use_batch_norm)

    out3 = conv1x1(out, 64, batch_norm=use_batch_norm)
    out3 = conv3x3(out3, 96, batch_norm=use_batch_norm)
    out3 = conv3x3(out3, 96, batch_norm=use_batch_norm)

    out4 = avg_pool_3x3(out)
    out4 = conv1x1(out4, 96, batch_norm=use_batch_norm)

    return concat(3, [out1, out2, out3, out4])


def avg_pool_3x3(out, stride=1, padding='SAME'):
    return tf.nn.avg_pool(out, ksize=[1, 3, 3, 1], strides=[1, stride, stride, 1], padding=padding)


def inceptionB(out):
    out1 = conv1x1(out, 384, batch_norm=use_batch_norm)

    out2 = conv1x1(out, 192, batch_norm=use_batch_norm)
    out2 = conv1x7(out2, 224, batch_norm=use_batch_norm)
    out2 = conv7x1(out2, 256, batch_norm=use_batch_norm)

    out3 = conv1x1(out, 192, batch_norm=use_batch_norm)
    out3 = conv7x1(out3, 192, batch_norm=use_batch_norm)
    out3 = conv1x7(out3, 224, batch_norm=use_batch_norm)
    out3 = conv7x1(out3, 224, batch_norm=use_batch_norm)
    out3 = conv1x7(out3, 256, batch_norm=use_batch_norm)

    out4 = avg_pool_3x3(out)
    out4 = conv1x1(out4, 128, batch_norm=use_batch_norm)

    return concat(3, [out1, out2, out3, out4])


def inceptionC(out):
    out1 = conv1x1(out, 256, batch_norm=use_batch_norm)

    out2 = conv1x1(out, 384, batch_norm=use_batch_norm)
    out21 = conv1x3(out2, 256, batch_norm=use_batch_norm)
    out22 = conv3x1(out2, 256, batch_norm=use_batch_norm)
    out2 = concat(3, [out21, out22])

    out3 = conv1x1(out, 384, batch_norm=use_batch_norm)
    out3 = conv3x1(out3, 448, batch_norm=use_batch_norm)
    out3 = conv1x3(out3, 512, batch_norm=use_batch_norm)
    out31 = conv1x3(out3, 256, batch_norm=use_batch_norm)
    out32 = conv3x1(out3, 256, batch_norm=use_batch_norm)
    out3 = concat(3, [out31, out32])

    out4 = avg_pool_3x3(out)
    out4 = conv1x1(out4, 256, batch_norm=use_batch_norm)

    return concat(3, [out1, out2, out3, out4])


def reductionA(out, k=0, l=0, m=0, n=0):
    out1 = conv3x3(out, n, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out2 = conv1x1(out, k, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, l, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, m, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out3 = max_pool_3x3(out, stride=2, padding="VALID")

    return concat(3, [out1, out2, out3])


def reductionB(out):
    out1 = conv1x1(out, 192, batch_norm=use_batch_norm)
    out1 = conv3x3(out1, 192, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out2 = conv1x1(out, 256, batch_norm=use_batch_norm)
    out2 = conv1x7(out2, 256, batch_norm=use_batch_norm)
    out2 = conv7x1(out2, 320, batch_norm=use_batch_norm)
    out2 = conv3x3(out2, 320, stride=2, padding="VALID", batch_norm=use_batch_norm)

    out3 = max_pool_3x3(out, stride=2, padding="VALID")

    return concat(3, [out1, out2, out3])


class InceptionV4(BaseImageClassificationModel):
    def __init__(self, width=299, height=299, channels=3, classes=10):
        super(InceptionV4, self).__init__()

        assert width == height, "width and height must be the same"

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        with tf.variable_scope("reshape1"):
            x = tf.reshape(x, [-1, width, height, channels])

        self._number_of_classes = classes

        max_w = 299
        min_w = 299 // 3

        with tf.variable_scope("resize1"):
            if width < min_w:
                x = tf.image.resize_images(x, [min_w, min_w])
            elif width == 299:
                pass  # do nothing
            else:
                x = tf.image.resize_images(x, [max_w, max_w])

        with tf.variable_scope("stem"):
            out = stem(x)

        for i in range(4):
            with tf.variable_scope("incA" + str(i)):
                out = inceptionA(out)
        with tf.variable_scope("redA"):
            out = reductionA(out, k=192, l=224, m=256, n=384)

        for i in range(7):
            with tf.variable_scope("incB" + str(i)):
                out = inceptionB(out)
        with tf.variable_scope("redB"):
            out = reductionB(out)

        for i in range(3):
            with tf.variable_scope("incC" + str(i)):
                out = inceptionC(out)

        a, b = out.get_shape().as_list()[1:3]
        out = tf.nn.avg_pool(out, ksize=[1, a, b, 1],
                             strides=[1, 1, 1, 1], padding="VALID")

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        with tf.variable_scope("reshape2"):
            out = tf.reshape(out, [-1, int(flatten_size)])

        with tf.variable_scope("fc1"):
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
        return "inception_bn"

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
