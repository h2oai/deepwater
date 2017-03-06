from deepwater.models import BaseImageClassificationModel

import tensorflow as tf
from tensorflow.python.ops import nn

from deepwater.models.nn import fc

class MultiLayerPerceptron(BaseImageClassificationModel):
    def __init__(self, width=28, height=28, channels=1, classes=10,
                 hidden_layers=[], input_dropout=0, dropout=[], activation_fn=nn.relu):
        super(MultiLayerPerceptron, self).__init__()

        self._number_of_classes = classes
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size], name="x")
        self._dropout_train_values = dropout
        dropout_shape = [len(dropout)]
        dropout_default = tf.zeros(dropout_shape, dtype=tf.float32)
        self._dropout_var = tf.placeholder_with_default(dropout_default,
                                                        dropout_shape,
                                                        name="dropout")

        self._inputs = x
        x = tf.nn.dropout(x, keep_prob=1.0 - input_dropout)
	
        for idx, h in enumerate(hidden_layers):
            with tf.variable_scope("fc%d" % idx):
                out = fc(x, h, activation_fn=activation_fn)

            if self._dropout_var.get_shape()[0] > idx:
                with tf.variable_scope("dropout%d" % idx):
                    out = tf.nn.dropout(out, keep_prob=1.0 - self._dropout_var[idx])

        with tf.variable_scope("fc%d" % len(hidden_layers)):
            out = fc(out, classes)

        self._logits = out
        if classes > 1:
            self._predictions = tf.nn.softmax(out)
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
