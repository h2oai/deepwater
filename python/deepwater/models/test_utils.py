import unittest
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from deepwater.datasets import cifar

import math

from deepwater import train


def generate_train_graph(model_class, optimizer_class,
                         width, height, channels, classes, add_summaries=False):
    graph = tf.Graph()
    with graph.as_default():

        is_train_var = tf.get_variable("is_training",
                               initializer=lambda *args, **kwds: False, shape=[],
                               trainable=False,
                               dtype=tf.bool)

        is_train = tf.placeholder_with_default(False, [])

        assign_train = is_train_var.assign(is_train)

        with tf.control_dependencies([assign_train, is_train_var.initialized_value()]):

            # 1. instantiate the model
            model = model_class(width, height, channels, classes)

            # 2. instantiate the optimizer
            optimizer = optimizer_class()

            # 3. instantiate the train wrapper
            train_strategy = train.ImageClassificationTrainStrategy(
                graph, model, optimizer, is_train, add_summaries=add_summaries)



        init = tf.global_variables_initializer()
        tf.add_to_collection("init", init.name)

    return train_strategy


class BaseImageClassificationTest(unittest.TestCase):
    pass


def CIFAR10_must_converge(name, model_class,
                          optimizer_class,
                          epochs=32,
                          batch_size=500,
                          initial_learning_rate=0.01,
                          summaries=False,
                          use_debug_session=False
                          ):
    train_strategy = generate_train_graph(
        model_class, optimizer_class, 32, 32, 3, 10)

    if summaries:
        filepath = "/tmp/%s/cifar10/train" % name
        train_writer = tf.summary.FileWriter(filepath)
        print("summaries: ", filepath)

    def train(epoch, dataset, batch_size, total, sess, summaries=False):
        average_loss = []
        average_error = []
        eye = np.eye(10)
        total_examples = 0

        def step_decay(epoch):
            initial_lrate = initial_learning_rate
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        learning_rate = step_decay(epoch)

        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)

            # one hot encode 
            y_batch = eye[label_batch]
            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: y_batch,
                train_strategy.learning_rate: learning_rate,
            }

            feed_dict.update(train_strategy.train_parameters)

            fetches = [train_strategy.optimize,
                       train_strategy.loss,
                       train_strategy.global_step,
                       train_strategy.predictions,
                       train_strategy.categorical_error,
                       ]

            _, loss, global_step, predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_loss.append(loss)
            average_error.append(error)

        if summaries:
            fetches = train_strategy.summary_op
            summary = sess.run(fetches, feed_dict=feed_dict)
            train_writer.add_summary(summary)

        return global_step, np.mean(average_loss), np.mean(average_error) * 100.

    def test(epoch, dataset, batch_size, total, sess, summaries=False):
        total_examples = 0
        average_error = []
        eye = np.eye(10)
        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)
            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: eye[label_batch],
            }

            fetches = [
                train_strategy.predictions,
                train_strategy.categorical_error,
            ]

            predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_error.append(error)

        # Add summaries
        if summaries:
            fetches = train_strategy.summary_op
            summary = sess.run(fetches, feed_dict=feed_dict)
            train_writer.add_summary(summary)

        return np.mean(average_error) * 100.0

    with tf.Session(graph=train_strategy.graph) as sess:
        tf.set_random_seed(12345678)
        sess.run(tf.get_collection('init')[0])

        if use_debug_session:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        dataset = cifar.read_data_sets('/tmp/deepwater/cifar10/', validation_size=0)

        print("computing initial test error ...")
        # test_error = test(0, dataset.test, batch_size,
        #                   dataset.test.num_examples, sess, summaries=summaries)
        #
        # print('initial test error:', test_error)

        for epoch in range(epochs):
            global_step, train_loss, train_error = train(epoch,
                                                         dataset.train, batch_size,
                                                         dataset.train.num_examples,
                                                         sess)
            test_error = test(epoch, dataset.test, batch_size,
                              dataset.test.num_examples, sess, summaries=summaries)

            print('epoch:', "%d/%d" % (epoch, epochs), 'step', global_step, 'train loss:', train_loss,
                  '% train error:', train_error,
                  '% test error:', test_error)

        train_writer.close()


def MNIST_must_converge(name,
                        model_class,
                        optimizer_class,
                        epochs=20,
                        batch_size=500,
                        initial_learning_rate=0.01,
                        summaries=False,
                        use_debug_session=False
                        ):
    train_strategy = generate_train_graph(
        model_class, optimizer_class, 28, 28, 1, 10, add_summaries=summaries)

    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    train_writer = tf.summary.FileWriter("/tmp/%s/train/%s" % (name, timestamp))

    def train(epoch, dataset, batch_size, total, sess):
        average_loss = []
        average_error = []
        eye = np.eye(10)
        total_examples = 0

        def step_decay(epoch):
            initial_lrate = initial_learning_rate
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        learning_rate = step_decay(epoch)

        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)
            # one hot encode 
            y_batch = eye[label_batch]
            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: y_batch,
                train_strategy.learning_rate: learning_rate,
            }

            feed_dict.update(train_strategy.train_parameters)

            fetches = [train_strategy.optimize,
                       train_strategy.loss,
                       train_strategy.global_step,
                       train_strategy.predictions,
                       train_strategy.categorical_error,
                       ]

            _, loss, global_step, predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_loss.append(loss)
            average_error.append(error)

            err = np.mean(average_error) * 100.0
            print("train: loss: %f err: %f lr: %f" % (np.mean(average_loss), err, learning_rate))

            if summaries and (total_examples % 10):
                fetches = train_strategy.summary_op
                summary = sess.run(fetches, feed_dict=feed_dict)
                train_writer.add_summary(summary)
                train_writer.flush()
                print("writing summaries")

        return global_step, np.mean(average_loss), np.mean(average_error) * 100.

    def test(epoch, dataset, batch_size, total, sess, summaries=True):
        total_examples = 0
        average_error = []
        eye = np.eye(10)
        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)

            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: eye[label_batch],
            }

            fetches = [
                train_strategy.predictions,
                train_strategy.categorical_error,
            ]

            predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_error.append(error)

            err = np.mean(average_error) * 100.0
            print("test err: %f" % err)

        return np.mean(average_error) * 100.0

    with tf.Session(graph=train_strategy.graph) as sess:
        tf.set_random_seed(12345678)
        sess.run(tf.get_collection('init')[0])

        if use_debug_session:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        dataset = read_data_sets('/tmp/deepwater/datasets/', validation_size=0)

        if not use_debug_session:
            print('computing initial test error')
            test_error = test(0, dataset.test, batch_size,
                              dataset.test.num_examples, sess, summaries=summaries)
            print('initial test error:', test_error)

        for epoch in range(epochs):
            global_step, train_loss, train_error = train(epoch,
                                                         dataset.train, batch_size,
                                                         dataset.train.num_examples,
                                                         sess)
            test_error = test(epoch, dataset.test, batch_size,
                              dataset.test.num_examples, sess, summaries=summaries)

            print('epoch:', "%d/%d" % (epoch, epochs), 'step', global_step, 'train loss:', train_loss,
                  '% train error:', train_error,
                  '% test error:', test_error)

        train_writer.close()
