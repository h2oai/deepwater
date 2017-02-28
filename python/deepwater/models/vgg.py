import tensorflow as tf

from deepwater.models import BaseImageClassificationModel

from deepwater.models.nn import conv3x3, fc, max_pool_3x3


class VGG16(BaseImageClassificationModel):

    def __init__(self, width=28, height=28, channels=1, classes=10):
        super(VGG16, self).__init__()

        assert width == height, "width and height must be the same"

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        with tf.variable_scope("reshape1"):
            x = tf.reshape(x, [-1, width, height, channels])

        self._number_of_classes = classes

        with tf.variable_scope("resize1"):
            if width < 224:
                x = tf.image.resize_images(x, [48, 48])
            elif width > 224:
                x = tf.image.resize_images(x, [224, 224])

        # 2 x 64
        with tf.variable_scope("conv1"):
            out = conv3x3(x, 64, stride=1)
        with tf.variable_scope("conv2"):
            out = conv3x3(out, 64, stride=1)
        with tf.variable_scope("conv3"):
            out = max_pool_3x3(out)

        # 2 x 128
        with tf.variable_scope("conv4"):
            out = conv3x3(out, 128, stride=1)
        with tf.variable_scope("conv5"):
            out = conv3x3(out, 128, stride=1)
            out = max_pool_3x3(out)

        # 3 x 256
        with tf.variable_scope("conv6"):
            out = conv3x3(out, 256, stride=1)
        with tf.variable_scope("conv7"):
            out = conv3x3(out, 256, stride=1)
        with tf.variable_scope("conv8"):
            out = conv3x3(out, 256, stride=1)
            out = max_pool_3x3(out)

        # 3 x 512
        with tf.variable_scope("conv9"):
            out = conv3x3(out, 512, stride=1)
        with tf.variable_scope("conv10"):
            out = conv3x3(out, 512, stride=1)
        with tf.variable_scope("conv11"):
            out = conv3x3(out, 512, stride=1)
            out = max_pool_3x3(out)

        # 512
        with tf.variable_scope("conv12"):
            out = conv3x3(out, 512, stride=1)
        with tf.variable_scope("conv13"):
            out = conv3x3(out, 512, stride=1)
        with tf.variable_scope("conv14"):
            out = conv3x3(out, 512, stride=1)
            out = max_pool_3x3(out)

        dims = out.get_shape().as_list()
        flatten_size = 1
        for d in dims[1:]:
            flatten_size *= d

        with tf.variable_scope("reshape2"):
            out = tf.reshape(out, [-1, int(flatten_size)])

        # fully connected
        with tf.variable_scope("fc1"):
            out = fc(out, [int(flatten_size), 4096])
            out = tf.nn.relu(out)
        with tf.variable_scope("fc2"):
            out = fc(out, [4096, 4096])
            out = tf.nn.relu(out)
        with tf.variable_scope("fc3"):
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
        return "vgg"

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
