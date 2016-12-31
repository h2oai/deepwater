import tensorflow as tf
from abc import ABCMeta, abstractproperty


def add_variable_summaries(var, name):
    """Attach summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


class BaseImageClassificationModel(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def number_of_classes(self):
        pass

    @abstractproperty
    def inputs(self):
        """ Returns the input tensor reference to the model """
        pass

    @abstractproperty
    def logits(self):
        """
        Returns the unscaled logits tensor reference of the model.
        This is typically the result of the last fully connected layer
        before applying softmax.
        """
        pass

    @abstractproperty
    def predictions(self):
        """
        This tensor returns a tensor containing the indices of the classes that
        the model predicts.
        """
        pass
