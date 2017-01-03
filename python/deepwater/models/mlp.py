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

                y = tf.matmul(x, w) + b
                y = tf.nn.relu(y)
                if len(dropout) < idx:
                    y = tf.nn.dropout(y, keep_prob=dropout[idx])
                x = y

        self._logits = y
        self._predictions = tf.nn.softmax(y)

    @property
    def name(self):
        return "MultiLayerPerceptron"

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
