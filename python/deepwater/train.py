import tensorflow as tf


class ImageClassificationTrainStrategy(object):
    """
    Wraps a image classification model and adds training operations.
    - uses cross_entropy as the default loss function

    """

    def __init__(self, graph, model, optimizer, add_summaries=False):
        self._graph = graph
        self._model = model
        self._labels = labels = tf.placeholder(tf.float32,
                                               [None, model.number_of_classes])
        logits = model.logits

        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits))

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

        max_norm_constraint = False
        if max_norm_constraint:
            for _, var in self._optimizer.grads_and_vars:
                var.assign(tf.clip_by_norm(var, 2.0))

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
            tf.summary.histogram(name, var)


def deepwater_image_classification_model(
        x, y, logits, loss, accuracy, optimizer):
    # This is required by the h2o tensorflow backend

    # train
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))

    train_op = optimizer.apply_gradients(grads_and_vars=grads)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)

    summary_op = tf.summary.merge()

    tf.add_to_collection("train", train_op)
    tf.add_to_collection("summary", summary_op)

    tf.add_to_collection("logits", y)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    tf.add_to_collection("init", init.name)

    meta = json.dumps({
        "inputs": {"batch_image_input": x.name, "categorical_labels": y.name},
        "outputs": {"categorical_logits": logits.name},
        "metrics": {"accuracy": accuracy.name, "total_loss": loss.name},
        "parameters": {"global_step": global_step.name},
    })
    tf.add_to_collection("meta", meta)

    return tf.train.export_meta_graph(saver_def=saver.as_saver_def())
