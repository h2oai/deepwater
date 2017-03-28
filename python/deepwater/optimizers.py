import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from abc import ABCMeta, abstractproperty, abstractmethod


class BaseOptimizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, loss):
        pass

    @abstractproperty
    def global_step(self):
        pass

    @abstractproperty
    def optimize_op(self):
        pass

    @abstractproperty
    def grads_and_vars(self):
        pass


class RMSPropOptimizer(BaseOptimizer):

    def __init__(self,
                 initial_learning_rate=1e-3,
                 initial_momentum=0.9,
                 decay=0.9,
                 epsilon=1.0
                 ):
        self._global_step = tf.Variable(0, name="global_step", trainable=False)
        self._learning_rate = \
            tf.placeholder_with_default(initial_learning_rate, [],
                                        name="learning_rate")
        self._momentum = \
            tf.placeholder_with_default(initial_momentum, [],
                                        name="momentum")

        self._decay = decay
        self._epsilon = epsilon

        self._optimizer = tf.train.RMSPropOptimizer(
            self._learning_rate, decay=decay, momentum=self._momentum, epsilon=epsilon)

        self._grads_and_vars = None
        self._optimize_op = None

    def apply(self, loss):
        trainable = tf.trainable_variables()
        self._grads_and_vars = self._optimizer.compute_gradients(loss, trainable)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self._optimize_op = self._optimizer.minimize(loss, global_step=self._global_step)


    @property
    def grads_and_vars(self):
        return self._grads_and_vars

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op


class MomentumOptimizer(BaseOptimizer):
    def __init__(self,
                 initial_learning_rate=1e-3,
                 initial_momentum=0.9,
                 ):
        self._global_step = tf.Variable(0, name="global_step", trainable=False)
        self._learning_rate = \
            tf.placeholder_with_default(initial_learning_rate, [],
                                        name="learning_rate")
        self._momentum = \
            tf.placeholder_with_default(initial_momentum, [],
                                        name="momentum")

        self._optimizer = tf.train.MomentumOptimizer(
            self._learning_rate, self._momentum)

        self._grads_and_vars = None
        self._optimize_op = None

    def apply(self, loss):
        trainable = tf.trainable_variables()
        self._grads_and_vars = self._optimizer.compute_gradients(loss, trainable)
        update_ops = tf.get_default_graph().get_collection(tf.GraphKeys.UPDATE_OPS)
        # print("lr %f  mom %f \n" %(self._learning_rate, self._momentum))
        with tf.get_default_graph().control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            # gradient_multipliers = {}
            # for v in trainable:
            #     print(v.name)
            #     if 'bias' not in v.name:
            #         gradient_multipliers[v.name] = 1.0/32

            self._optimize_op = tf.contrib.layers.optimize_loss(loss,
                                                                tf.contrib.framework.get_global_step(),
                                                                self._learning_rate,
                                                                optimizer=lambda lr: tf.train.MomentumOptimizer(self._learning_rate, momentum=self._momentum),
                                                                #gradient_multipliers=gradient_multipliers,
                                                                clip_gradients=10.0)
                                                                # )

    @property
    def grads_and_vars(self):
        return self._grads_and_vars

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self,
                 initial_learning_rate=0.1,
                 ):
        self._global_step = tf.Variable(0, name="global_step", trainable=False)
        self._learning_rate = \
            tf.placeholder_with_default(initial_learning_rate, [],
                                        name="learning_rate")

        self._optimizer = tf.train.GradientDescentOptimizer(
            self._learning_rate)
        self._grads_and_vars = None
        self._optimize_op = None

    def apply(self, loss):
        trainable = tf.trainable_variables()
        self._grads_and_vars = self._optimizer.compute_gradients(loss, trainable)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self._optimize_op = self._optimizer.minimize(loss, global_step=self._global_step)

    @property
    def grads_and_vars(self):
        return self._grads_and_vars

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op

class AdamOptimizer(BaseOptimizer):
    def __init__(self,
                 initial_learning_rate=1e-3,
                 initial_momentum=0.9,
                 ):
        self._global_step = tf.Variable(0, name="global_step", trainable=False)
        self._learning_rate = \
            tf.placeholder_with_default(initial_learning_rate, [],
                                        name="learning_rate")
        self._momentum = \
            tf.placeholder_with_default(initial_momentum, [],
                                        name="momentum")

        self._optimizer = tf.train.AdamOptimizer(
            self._learning_rate)

        self._grads_and_vars = None
        self._optimize_op = None

    def apply(self, loss):
        trainable = tf.trainable_variables()
        self._grads_and_vars = self._optimizer.compute_gradients(loss, trainable)
        update_ops = tf.get_default_graph().get_collection(ops.GraphKeys.UPDATE_OPS)
        with tf.get_default_graph().control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self._optimize_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

    @property
    def grads_and_vars(self):
        return self._grads_and_vars

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op

class DefaultOptimizer(BaseOptimizer):
    def __init__(self,
                 initial_learning_rate=0.01,
                 num_steps_per_decay=1000,
                 decay_rate=0.96):
        self._global_step = tf.Variable(0, name="global_step", trainable=False)

        self._learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            self._global_step,
            num_steps_per_decay,
            decay_rate,
            staircase=True)

        self._optimizer = tf.train.AdagradOptimizer(
            self._learning_rate)

        self._grads_and_vars = None
        self._optimize_op = None

    def apply(self, loss):
        trainable = tf.trainable_variables()
        self._grads_and_vars = self._optimizer.compute_gradients(loss, trainable)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self._optimize_op = self._optimizer.minimize(loss, global_step=self._global_step)

        for _, var in self._grads_and_vars:
            print(var.name)
            var.assign(tf.clip_by_norm(var, 2.0))

    @property
    def grads_and_vars(self):
        return self._grads_and_vars

    @property
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op
