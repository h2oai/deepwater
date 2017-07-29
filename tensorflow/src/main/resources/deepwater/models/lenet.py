import tensorflow as tf

from deepwater.models import BaseImageClassificationModel
from deepwater.models.nn import max_pool_2x2, conv, fc


class LeNet(BaseImageClassificationModel):
    def __init__(self, width, height, channels, classes, number_of_gpus=4):
        super(LeNet, self).__init__()

        # activation_function = "tanh"
        activation_function = "relu"

        self._number_of_classes = classes

        size = width * height * channels

        x = tf.placeholder(tf.float32, [None, size], name="x")

        self._inputs = x

        with tf.variable_scope("reshape1"):
            x_image = tf.reshape(x, [-1, width, height, channels],
                                 name="input_reshape")

        if number_of_gpus <= 1:
            gpus = 1
        else:
            gpus = number_of_gpus

        if gpus > 1:
            x_image_arr = tf.split(x_image, gpus, axis=0)
        else:
            x_image_arr = [x_image]
        classes_arr = [classes] * gpus

        logits_arr = [0.0] * gpus

        for gpu in range(gpus):
            with tf.device('/gpu:' + str(gpu)):

                with tf.variable_scope("conv1_" + str(gpu)):
                    out = conv(x_image_arr[gpu], 5, 5, 20, activation=activation_function)
                    out = max_pool_2x2(out)

                with tf.variable_scope("conv2_" + str(gpu)):
                    out = conv(out, 5, 5, 50, activation=activation_function)
                    out = max_pool_2x2(out)

                dims = out.get_shape().as_list()
                flatten_size = 1
                for d in dims[1:]:
                    flatten_size *= d

                with tf.variable_scope("reshape2_" + str(gpu)):
                    flatten = tf.reshape(out, [-1, int(flatten_size)])

                with tf.variable_scope("fc1_" + str(gpu)):
                    out = fc(flatten, 500)
                    if activation_function == "relu":
                        out = tf.nn.relu(out)
                    elif activation_function == "tanh":
                        out = tf.nn.tanh(out)

                with tf.variable_scope("fc2_" + str(gpu)):
                    logits_arr[gpu] = fc(out, classes_arr[gpu])

        self._logits = tf.concat(logits_arr, 0)

        # print(self._logits.graph.as_graph_def())

        if classes > 1:
            self._predictions = tf.nn.softmax(self._logits)
        else:
            self._predictions = self._logits

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
