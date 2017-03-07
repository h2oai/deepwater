import tensorflow as tf

class ImageClassificationTrainStrategy(object):
    """
    Wraps a image classification model and adds training operations.
    - uses cross_entropy as the default loss function

    """

    def __init__(self, graph, model, optimizer, batch_size, weight_decay=1e-6, add_summaries=False):
        self._graph = graph
        self._model = model
        self._labels = labels = tf.placeholder(tf.float32,
                                               [None, model.number_of_classes])

        self._batch_size = batch_size

        logits = model.logits

        self._l2_loss = None

        # weight regularization
        trainable_vars = tf.trainable_variables()
        l2 = tf.contrib.layers.l2_regularizer(weight_decay)
        self._l2_loss = tf.add_n([l2(v) for v in trainable_vars
                                  if 'bias' not in v.name])

        # Classification model
        if model.number_of_classes > 1:
            if weight_decay > 0.0:
                self._loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits) + self._l2_loss)
            else:
                self._loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits))
        # Regression model
        else:
            if weight_decay > 0.0:
                # Add weight decay
                self._loss = tf.losses.mean_squared_error(labels, logits) + self._l2_loss
            else:
                self._loss = tf.losses.mean_squared_error(labels, logits)

        a = tf.argmax(model.predictions, 1)
        b = tf.argmax(self._labels, 1)
        correct_predictions = tf.equal(a, b)
        wrong_predictions = tf.not_equal(a, b)

        self._accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                tf.float32))

        self._error = tf.reduce_mean(tf.cast(wrong_predictions,
                                             tf.float32))

        self._optimizer = optimizer
        self._optimizer.apply(self._loss)

        if add_summaries:
            self._add_summaries()
        self._summary_op = tf.summary.merge_all()

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def train_parameters(self):
        return self._model.train_dict

    @property
    def learning_rate(self):
        return self._optimizer._learning_rate

    @property
    def global_step(self):
        """
        Returns the tensor that references the inputs of the model.
        """
        return self._optimizer.global_step

    @property
    def inputs(self):
        """
        Returns the tensor that references the inputs of the model.
        """
        return self._model.inputs

    @property
    def labels(self):
        """
        Returns the tensor that references the target labels used during
        training.
        """
        return self._labels

    @property
    def predictions(self):
        """ Returns the tensor containing the scaled predictins values """
        return self._model.predictions

    @property
    def logits(self):
        """ Returns the tensor containing the logits value """
        return self._model.logits

    @property
    def categorical_error(self):
        """ Returns the tensor containing the categorical error"""
        return self._error

    @property
    def accuracy(self):
        """ Returns the tensor containing the accuracy"""
        return self._accuracy

    @property
    def loss(self):
        """ Returns the tensor containing the loss value """
        return self._loss

    @property
    def optimize(self):
        """
        Returns a reference to the optimization operation
        """
        return self._optimizer.optimize_op

    @property
    def graph(self):
        """
        Returns a reference to the optimization operation
        """
        return self._graph

    @property
    def batch_size(self):
        return self._batch_size

    def _add_summaries(self):
        tf.summary.scalar("loss", self.loss)

        for grad, var in self._optimizer.grads_and_vars:
            self._add_variable_summaries(var, var.name)
            self._add_variable_summaries(grad, var.name + '/gradient')

    def _add_variable_summaries(self, var, name):
        """Attach summaries to a Tensor."""
        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            # tf.summary.scalar('mean/' + name, mean)
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # tf.summary.scalar('sttdev/' + name, stddev)
            # tf.summary.scalar('max/' + name, tf.reduce_max(var))
            # tf.summary.scalar('min/' + name, tf.reduce_min(var))
            name = name.replace(":", "_")
            # tf.summary.histogram(name, var)
