import tensorflow as tf

from deepwater.models import BaseImageClassificationModel

from deepwater.models.nn import weight_variable, bias_variable, max_pool_3x3, max_pool_2x2, fc


def conv1x1(x, filters, **kwds):
    return conv(x, 1, 1, filters, **kwds)


def conv3x3(x, filters, **kwds):
    return conv(x, 3, 3, filters, **kwds)


def conv1x3(x, filters, **kwds):
    return conv(x, 1, 3, filters, **kwds)


def conv3x1(x, filters, **kwds):
    return conv(x, 3, 1, filters, **kwds)


def conv1x7(x, filters, **kwds):
    return conv(x, 1, 7, filters, **kwds)


def conv7x1(x, filters, **kwds):
    return conv(x, 7, 1, filters, **kwds)


def conv(x, w, h, filters, stride=1, padding="SAME"):
    channels = x.get_shape().as_list()[3]

    kernel_shape = [w, h, channels, filters]
    kernel = weight_variable(kernel_shape, "kernel")
    b = bias_variable([kernel_shape[-1]], "bias")
    x = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(x + b)


def stem(x):
    """
    Stem fo the pure InceptionV4 and Inception-ResNet-V2
    """
    out = conv3x3(x, 32, stride=2, padding="VALID")
    out = conv3x3(out, 32, stride=1, padding="VALID")
    out = conv3x3(out, 64, stride=1, padding="SAME")

    out1 = max_pool_3x3(out, stride=2, padding="VALID")
    out2 = conv3x3(out, 96, stride=2, padding="VALID")

    out = tf.concat(3, [out1, out2])

    out1 = conv1x1(out, 64)
    out1 = conv3x3(out1, 96, padding="VALID")

    out2 = conv1x1(out, 64)
    out2 = conv1x7(out2, 64)
    out2 = conv7x1(out2, 64)
    out2 = conv3x3(out2, 96, padding="VALID")

    out = tf.concat(3, [out1, out2])

    out1 = conv3x3(out, 192, stride=2, padding="VALID")
    out2 = max_pool_2x2(out, stride=2, padding="VALID")

    return tf.concat(3, [out1, out2])


def inceptionA(out):
    out1 = avg_pool_3x3(out)
    out1 = conv1x1(out1, 96)

    out2 = conv1x1(out, 96)

    out3 = conv1x1(out, 64)
    out3 = conv3x3(out3, 96)

    out4 = conv1x1(out, 64)
    out4 = conv3x3(out4, 96)
    out4 = conv3x3(out4, 96)

    return tf.concat(3, [out1, out2, out3, out4])


def avg_pool_3x3(out):
    return tf.nn.avg_pool(out, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def inceptionB(out):
    out1 = avg_pool_3x3(out)
    out1 = conv1x1(out1, 128)

    out2 = conv1x1(out, 384)

    out3 = conv1x1(out, 192)
    out3 = conv1x7(out3, 224)
    out3 = conv7x1(out3, 256)

    out4 = conv1x1(out, 192)
    out4 = conv1x7(out4, 192)
    out4 = conv7x1(out4, 224)
    out4 = conv1x7(out4, 224)
    out4 = conv7x1(out4, 256)

    return tf.concat(3, [out1, out2, out3, out4])


def inceptionC(out):
    out1 = avg_pool_3x3(out)
    out1 = conv1x1(out1, 256)

    out2 = conv1x1(out, 256)

    out3 = conv1x1(out, 384)
    out31 = conv3x1(out3, 256)
    out32 = conv1x3(out3, 256)

    out4 = conv1x1(out, 384)
    out4 = conv1x3(out4, 448)
    out4 = conv3x1(out4, 512)
    out41 = conv3x1(out4, 256)
    out42 = conv1x3(out4, 256)

    return tf.concat(3, [out1, out2, out31, out32, out41, out42])


def reductionA(out, k=0, l=0, m=0, n=0):
    out1 = max_pool_3x3(out, stride=2, padding="VALID")

    out2 = conv3x3(out, n, stride=2, padding="VALID")

    out3 = conv1x1(out, k)
    out3 = conv3x3(out3, l)
    out3 = conv3x3(out3, m, stride=2, padding="VALID")

    return tf.concat(3, [out1, out2, out3])


def reductionB(out):
    out1 = max_pool_3x3(out, stride=2, padding="VALID")

    out2 = conv1x1(out, 192)
    out2 = conv3x3(out2, 192, stride=2, padding="VALID")

    out3 = conv1x1(out, 256)
    out3 = conv1x7(out3, 256)
    out3 = conv7x1(out3, 256)
    out3 = conv1x7(out3, 320)
    out3 = conv3x3(out3, 320, stride=2, padding="VALID")

    return tf.concat(3, [out1, out2, out3])


class InceptionV4(BaseImageClassificationModel):
    def __init__(self, width=299, height=299, channels=3, classes=10):
        super(InceptionV4, self).__init__()

        self._dropout_var = tf.placeholder_with_default(0.0,
                                                        [],
                                                        name="dropout")

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

        out = stem(x)
        for _ in range(4):
            out = inceptionA(out)
        out = reductionA(out, k=192, l=224, m=256, n=384)

        for _ in range(7):
            out = inceptionB(out)
        out = reductionB(out)

        for _ in range(3):
            out = inceptionC(out)

        a, b = out.get_shape().as_list()[1:3]

        out = tf.nn.avg_pool(out, ksize=[1, a, b, 1],
                             strides=[1, 1, 1, 1], padding="VALID")

        out = tf.nn.dropout(out, keep_prob=1.0-self._dropout_var)

        flatten_size = 1 * 1 * 1536

        out = tf.reshape(out, [-1, flatten_size])
        out = fc(out, [flatten_size, classes])

        self._logits = out

        if classes > 1:
            self._predictions = tf.nn.softmax(self._logits)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {
            self._dropout_var: 1.0 - 0.8
        }

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


def inceptionResNetA(out):
    out = tf.nn.relu(out)

    out1 = conv1x1(out, 32)

    out2 = conv1x1(out, 32)
    out2 = conv3x3(out2, 32)

    out3 = conv1x1(out, 32)
    out3 = conv3x3(out3, 48)
    out3 = conv3x3(out3, 64)

    out4 = conv1x1()

    total = out
    return tf.nn.relu(total)


def inceptionResNetB(out):
    pass


def inceptionResNetC(out):
    pass


class InceptionResNetV2(BaseImageClassificationModel):

    def __init__(self, width=299, height=299, channels=3, classes=10):
        super(InceptionResNetV2, self).__init__()

        self._dropout_var = tf.placeholder_with_default(0.0,
                                                        [],
                                                        name="dropout")

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

        out = stem(x)
        for _ in range(5):
            out = inceptionResNetA(out)
        out = reductionA(out, k=256, l=256, m=384, n=384)

        for _ in range(10):
            out = inceptionResNetB(out)
        out = reductionB(out)

        for _ in range(5):
            out = inceptionResNetC(out)

        a, b = out.get_shape().as_list()[1:3]

        out = tf.nn.avg_pool(out, ksize=[1, a, b, 1],
                             strides=[1, 1, 1, 1], padding="VALID")

        out = tf.nn.dropout(out, keep_prob=1.0-self._dropout_var)

        flatten_size = 1 * 1 * 1536

        out = tf.reshape(out, [-1, flatten_size])
        out = fc(out, [flatten_size, classes])

        self._logits = out

        if classes > 1:
            self._predictions = tf.nn.softmax(self._logits)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {
            self._dropout_var: 1.0 - 0.8
        }

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