import tensorflow as tf
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

class MomentumOptimizer(BaseOptimizer):

    def __init__(self,
                 initial_learning_rate=0.1,
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

    def apply(self, loss):
        trainables = tf.trainable_variables()
        grads_and_vars = self._optimizer.compute_gradients(loss, trainables)
        self._grads_and_vars = grads_and_vars
        self._optimize_op = self._optimizer.apply_gradients(grads_and_vars,
                                                            global_step=self._global_step)

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
    def global_step(self):
        return self._global_step

    @property
    def optimize_op(self):
        return self._optimize_op

class GradientDescentOptimizer(BaseOptimizer):

    def __init__(self,
                 initial_learning_rate=0.95,
                 initial_momentum=0.95,
                 ):

        self._global_step = tf.Variable(0, name="global_step", trainable=False)
        self._learning_rate = \
        tf.placeholder_with_default(initial_learning_rate, [],
                name="learning_rate")

        self._optimizer = tf.train.GradientDescentOptimizer(
            self._learning_rate)

    def apply(self, loss):
        trainables = tf.trainable_variables()
        grads_and_vars = self._optimizer.compute_gradients(loss, trainables)
        self._grads_and_vars = grads_and_vars
        self._optimize_op = self._optimizer.apply_gradients(grads_and_vars,
                                                            global_step=self._global_step)


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

    def apply(self, loss):
        trainables = tf.trainable_variables()
        grads_and_vars = self._optimizer.compute_gradients(loss, trainables)
        self._grads_and_vars = grads_and_vars
        self._optimize_op = self._optimizer.apply_gradients(grads_and_vars,
                                                            global_step=self._global_step)
        
        for _, var in grads_and_vars:
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
