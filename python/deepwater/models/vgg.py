import math 

from deepwater.models import BaseImageClassificationModel
import tensorflow as tf

def weight_variable(shape, name):
    # Delving deep into Rectifier
    n = shape[0] 
    factor=2.0 
    stddev=math.sqrt(1.3 * factor/n)

    initialization = tf.truncated_normal(
        shape, mean=0.0, stddev=stddev)

    return tf.Variable(initialization, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial)
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def block(x, kernel_shape):
    kernel = weight_variable(kernel_shape, "kernel")
    b = bias_variable([kernel_shape[-1]], "bias")
    out = conv2d(x, kernel) + b
    return tf.nn.relu(out)

def fc(x, shape):
    W = weight_variable(shape, "weight")
    b = bias_variable([shape[-1]], "bias")
    out = tf.matmul(x, W) + b
    return tf.nn.relu(out)


class VGG(BaseImageClassificationModel):

    def __init__(self, width=28, height=28, channels=1, classes=10, 
            hidden_layers=[], dropout=[]):
        super(VGG, self).__init__()

        assert width == height, "width and height must be the same"

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._inputs = x

        x = tf.reshape(x, [-1, width, height, channels])

        self._number_of_classes = classes

        input_width = width

        # 64
        out = block(x, [3, 3, channels, 64])
        out = block(out, [3, 3, 64, 64])
        if (width / 2 > 3): 
            input_width /= 2
            out = max_pool_2x2(out) 

        # 128
        out = block(out, [3, 3, 64, 128])
        out = block(out, [3, 3, 128, 128])
        if (width / 4 > 3): 
            input_width /= 2
            out = max_pool_2x2(out)  

        # 256
        out = block(out, [3, 3, 128, 256])
        out = block(out, [3, 3, 256, 256])
        if (width / 8 > 3): 
            input_width /= 2
            out = max_pool_2x2(out)

        # 512
        out = block(out, [3, 3, 256, 512])
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        if (width / 16 > 3): 
            input_width /= 2
            out = max_pool_2x2(out)

        # 512
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        out = block(out, [3, 3, 512, 512])
        if (width / 64 > 1): 
            input_width /= 2
            out = max_pool_2x2(out)

        flatten_size = input_width * input_width * 512

        out = tf.reshape(out, [-1, int(flatten_size)])

        print(out.get_shape())

        # fully connected
        out = fc(out, [int(flatten_size), 4096])
        out = fc(out, [4096, 4096])
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

