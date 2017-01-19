import math 

from deepwater.models import BaseImageClassificationModel
import tensorflow as tf


class MultiLayerPerceptron(BaseImageClassificationModel):

    def __init__(self, width=28, height=28, channels=1, classes=10, 
            hidden_layers=[], dropout=[]):
        super(MultiLayerPerceptron, self).__init__()

        self._number_of_classes = classes
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._dropout_train_values = dropout
        dropout_shape = [len(dropout)]
        droupout_default = tf.zeros(dropout_shape, dtype=tf.float32)
        self._dropout_var = tf.placeholder_with_default(droupout_default,
                                                  dropout_shape,
                                                  name="dropout")

        self._inputs = x
        if not hidden_layers:
            hidden_layers = [size, classes]
        else:
            hidden_layers = [size] + hidden_layers[:] + [classes]

        for idx, (h1, h2) in enumerate(zip(hidden_layers, hidden_layers[1:])):

            with tf.variable_scope("fc%d" % idx):
                # Delving deep into Rectifier
                n = h1
                factor=2.0 
                stddev=math.sqrt(1.3 * factor/n)

                initialization = tf.truncated_normal(
                    [h1, h2], mean=0.0, stddev=stddev)

                w = tf.Variable(initialization, name="W")
                b = tf.Variable(tf.zeros([h2]), name="b")

                y1 = tf.matmul(x, w) + b
                y2 = tf.nn.relu(y1)

            if self._dropout_var.get_shape()[0] > idx:
                with tf.variable_scope("dropout%d" % idx):
                    y2 = tf.nn.dropout(y2, keep_prob=1.0-self._dropout_var[idx])

            x = y2

        self._logits = y1
        if classes > 1:
            self._predictions = tf.nn.softmax(y1)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {
            self._dropout_var: self._dropout_train_values
        }

    @property
    def name(self):
        return "mlp"

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
