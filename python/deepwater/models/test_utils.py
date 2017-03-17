import unittest

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import time
from datetime import datetime

import numpy as np
import math

import scipy.misc

from deepwater.datasets import cifar
from deepwater import train
import os

import random

print_logs = False

def generate_train_graph(model_class, optimizer_class,
                         width, height, channels, classes, add_summaries=False):
    graph = tf.Graph()
    with graph.as_default():
        tf.placeholder(tf.bool, name="global_is_training")
        # 1. instantiate the model
        model = model_class(width, height, channels, classes)

        batch_size = tf.placeholder(tf.float32, [], name="batch_size")

        # 2. instantiate the optimizer
        optimizer = optimizer_class()

        # 3. instantiate the train wrapper
        train_strategy = train.ImageClassificationTrainStrategy(
            graph, model, optimizer, batch_size, add_summaries=add_summaries)

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
                train_strategy.batch_size: batch_size,
                "global_is_training:0": True,
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
                train_strategy.batch_size: batch_size,
                "global_is_training:0": False,
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
        if summaries:
            train_writer.close()

def MNIST_must_converge(name,
                        model_class,
                        optimizer_class,
                        epochs=50,
                        batch_size=32,
                        initial_learning_rate=0.001,
                        summaries=False,
                        use_debug_session=False
                        ):
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

        learning_rate = initial_learning_rate#step_decay(epoch)

        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)
            # one hot encode 
            y_batch = eye[label_batch]
            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: y_batch,
                train_strategy.learning_rate: learning_rate,
                train_strategy.batch_size: batch_size,
                "global_is_training:0": True
            }

            feed_dict.update(train_strategy.train_parameters)

            fetches = [train_strategy.optimize,
                       train_strategy.loss,
                       train_strategy.global_step,
                       train_strategy.predictions,
                       train_strategy.categorical_error,
                       ]

            if sess.should_stop():
                return global_step, np.mean(average_loss), np.mean(average_error) * 100.

            #if not sess.should_stop():
            _, loss, global_step, predictions, error = sess.run(fetches, feed_dict=feed_dict)

            average_loss.append(loss)
            average_error.append(error)

            err = np.mean(average_error) * 100.0

            if summaries and (total_examples % 10):
                fetches = train_strategy.summary_op
                summary = sess.run(fetches, feed_dict=feed_dict)
                train_writer.add_summary(summary)
                train_writer.flush()
                print("writing summaries")

        return global_step, np.mean(average_loss), err

    def test(dataset, batch_size, total, sess):
        total_examples = 0
        average_error = []
        eye = np.eye(10)

        while total_examples <= total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)

            feed_dict = {
                train_strategy.inputs: x_batch,
                train_strategy.labels: eye[label_batch],
                train_strategy.batch_size: batch_size,
                "global_is_training:0": False,
            }

            fetches = [
                train_strategy.predictions,
                train_strategy.categorical_error,
            ]

           # if not sess.should_stop():
            predictions, error = sess.run(fetches, feed_dict=feed_dict)

            average_error.append(error)

        err = np.mean(average_error) * 100.0
        print("test err: %f" % err)

        return err


    # run test on test set at end just before closing the session
    class TestAtEnd(tf.train.StopAtStepHook):
        def __init__(self, last_step):
            tf.train.StopAtStepHook.__init__(self, last_step=last_step)

        def end(self, session):
            print("computing final test error")
            test(dataset.test, batch_size, dataset.test.num_examples, session)

    print("Testing %s" % name)

    train_strategy = generate_train_graph(
        model_class, optimizer_class, 28, 28, 1, 10, add_summaries=summaries)

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    train_writer = tf.summary.FileWriter("/tmp/%s/train/%s" % (name, timestamp))

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1

        def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            return tf.train.SessionRunArgs(train_strategy.loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            duration = time.time() - self._start_time
            loss_value = run_values.results
            if self._step % 10 == 0 and print_logs:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
                print (format_str % (datetime.now(), self._step, loss_value,
                                     examples_per_sec, sec_per_batch))

    with train_strategy.graph.as_default():
        dataset = read_data_sets('/tmp/deepwater/datasets/', validation_size=0)
        checkpoint_directory="/tmp/checkpoint"
        checkpoint_file=checkpoint_directory + "/checkpoint"
        if os.path.isfile(checkpoint_file):
            os.remove(checkpoint_file)
        start_time = time.time()
        with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=checkpoint_directory,
                    hooks=[ TestAtEnd(epochs*dataset.train.num_examples), _LoggerHook() ],
                    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            epoch = 0

            tf.set_random_seed(12345678)

            if use_debug_session:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


            if not use_debug_session:
                print('computing initial test error')
                test_error = test(dataset.test, batch_size, dataset.test.num_examples, sess)
                print('initial test error: %f' % (test_error))

            while not sess.should_stop() and epoch < epochs:
                print('epoch: %d' % (epoch))
                epoch += 1
                global_step, train_loss, train_error = train(epoch,
                                                             dataset.train, batch_size,
                                                             dataset.train.num_examples,
                                                             sess)
                print("train: avg loss %f  error %f" % (train_loss, train_error))

            train_writer.close()

        elapsed_time = time.time() - start_time
        print("time %.2f s\n" % elapsed_time)

trained_global = 0

def cat_dog_mouse_must_converge(name,
                        model_class,
                        optimizer_class,
                        epochs=20,
                        batch_size=500,
                        initial_learning_rate=0.01,
                        summaries=False
                        ):
    def create_batches(batch_size, images, labels):
        images_batch = []
        labels_batch = []

        for img in images:
            imread = scipy.misc.imresize(scipy.misc.imread(img), [299, 299]).reshape(1,299*299*3)
            images_batch.append(imread)

        modulus = len(images_batch) % batch_size

        print(len(images_batch))

        if modulus != 0:
            i = 0
            while len(images_batch) % batch_size != 0:
                imread = scipy.misc.imresize(scipy.misc.imread(images[i]), [299, 299]).reshape(1,299*299*3)
                images_batch.append(imread)
                i += 1

        print(len(images_batch))

        for label in labels:
            labels_batch.append(label)

        if modulus != 0:
            i = 0
            while len(labels_batch) % batch_size != 0:
                labels_batch.append(labels[i])
                i += 1

        labels_batch = np.asarray(labels_batch)

        while (True):
            for i in range(0,len(images_batch),batch_size):
                b = random.sample(range(0, len(images_batch)), batch_size)
                yield( [ images_batch[i] for i in b ], [ labels_batch[i] for i in b ])

    def train(batch_generator, sess):

        global trained_global
        trained = 0

        learning_rate = initial_learning_rate

        while trained + batch_size <= 288:
            batched_images, batched_labels = batch_generator.next()
            images = np.asarray(batched_images).reshape(batch_size, 299*299*3)
            labels = eye[batched_labels]

            trained += batch_size

            feed_dict = {
                train_strategy.inputs: images,
                train_strategy.labels: labels,
                train_strategy.learning_rate: learning_rate,
                "global_is_training:0": True,
            }

            feed_dict.update(train_strategy.train_parameters)

            fetches = [train_strategy.optimize]

            sess.run(fetches, feed_dict=feed_dict)

        trained_global += trained
        print('trained %d' % (trained_global))

    def test(batch_generator, sess):

        global trained_global
        trained = 0

        average_loss = []
        average_error = []

        learning_rate = initial_learning_rate

        while trained + batch_size <= 288:
            batched_images, batched_labels = batch_generator.next()
            images = np.asarray(batched_images).reshape(batch_size, 299*299*3)
            labels = eye[batched_labels]

            trained += batch_size

            feed_dict = {
                train_strategy.inputs: images,
                train_strategy.labels: labels,
                train_strategy.learning_rate: learning_rate,
                "global_is_training:0": False,
            }

            feed_dict.update(train_strategy.train_parameters)

            fetches = [train_strategy.loss,
                       train_strategy.global_step,
                       train_strategy.predictions,
                       train_strategy.categorical_error,
                       ]

            loss, global_step, predictions, error = sess.run(fetches, feed_dict=feed_dict)

            average_loss.append(loss)
            average_error.append(error)

        return 0, np.mean(average_loss), np.mean(average_error) * 100.

    def read_labeled_image_list(image_list_file):
        f = open(image_list_file, 'r')
        filenames = []
        labels = []
        label_domain = ['cat', 'dog', 'mouse']
        for line in f:
            filename, label = line[:-1].split(' ')
            filenames.append(filename)
            labels.append(label_domain.index(label))
        return filenames, labels

    train_strategy = generate_train_graph(
        model_class, optimizer_class, 299, 299, 3, 3, add_summaries=summaries)

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1

        def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            return tf.train.SessionRunArgs(train_strategy.loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            duration = time.time() - self._start_time
            loss_value = run_values.results
            if self._step % 10 == 0 and print_logs:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), self._step, loss_value,
                                     examples_per_sec, sec_per_batch))

    with train_strategy.graph.as_default():
        epoch = 0

        tf.set_random_seed(12345678)

        # Load the data
        image, labels = read_labeled_image_list("bigdata/laptop/deepwater/imagenet/cat_dog_mouse.csv")

        batch_generator = create_batches(batch_size, image, labels)
        start_time = time.time()
        with tf.train.MonitoredTrainingSession(hooks=[ _LoggerHook() ]) as sess:

            # sess.run(tf.global_variables_initializer())

            for _ in range(epochs):
                epoch += 1
                eye = np.eye(3)
                train(batch_generator, sess)

                global_step, train_loss, train_error = test(batch_generator, sess)

                print('epoch:', "%d/%d" % (epoch, epochs), 'step', global_step, 'test loss:', train_loss,
                      '% test error:', train_error)

            # _, train_loss, train_error = train(batch_generator, sess)
            # print('final train error: %f' % (train_error))
            # return train_error

        elapsed_time = time.time() - start_time
        print("time %.2f s\n" % elapsed_time)