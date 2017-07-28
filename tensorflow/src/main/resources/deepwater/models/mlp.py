from deepwater.models import BaseImageClassificationModel

import tensorflow as tf

from tensorflow.python.ops import nn

from deepwater.models.nn import fc


class MultiLayerPerceptron(BaseImageClassificationModel):
    def __init__(self, width=28, height=28, channels=1, classes=10,
                 hidden_layers=[], dropout=[], activation_fn=nn.relu):
        super(MultiLayerPerceptron, self).__init__()

        self._number_of_classes = classes
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size], name="x")

        # Hidden dropout
        dropout_default = tf.zeros([2], dtype=tf.float32)
        self._hidden_dropout = tf.placeholder_with_default(dropout_default,
                                                           [None],
                                                           name="hidden_dropout")

        # Input dropout
        self._input_dropout = tf.placeholder_with_default(tf.constant(0.0, dtype=tf.float32),
                                                          [],
                                                          name="input_dropout")

        # Activations
        self._activations = tf.placeholder_with_default([0, 0],
                                                        [None],
                                                        name="activations")

        self._inputs = x
        # out = tf.nn.dropout(x, keep_prob=tf.constant(1.0, dtype=tf.float32) - self._input_dropout)
        out = x

        for idx, h in enumerate(hidden_layers):
            with tf.variable_scope("fc%d" % idx):
                if 1 == self._activations[idx]:
                    activation_fn = nn.tanh
                else:
                    activation_fn = nn.relu
                out = fc(out, h, activation_fn=activation_fn)

            if self._hidden_dropout.get_shape()[0] > idx:
                with tf.variable_scope("dropout%d" % idx):
                    out = tf.nn.dropout(out, keep_prob=1.0 - self._hidden_dropout[idx])

        with tf.variable_scope("fc%d" % len(hidden_layers)):
            out = fc(out, classes)

        self._logits = out
        if classes > 1:
            self._predictions = tf.nn.softmax(out)
        else:
            self._predictions = self._logits

    @property
    def train_dict(self):
        return {}

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

    @property
    def hidden_dropout(self):
        return self._hidden_dropout

    @property
    def input_dropout(self):
        return self._input_dropout

    @property
    def activations(self):
        return self._activations
