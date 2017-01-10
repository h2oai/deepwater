import unittest

import numpy as np
import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import math

from deepwater.models import mlp
from deepwater import optimizers
from deepwater import train 


def generate_train_graph(modelClass, optimizerClass,
                         width, height, channels, classes):
    graph = tf.Graph()
    with graph.as_default():
        # 1. instantiate the model
        model = modelClass(width, height, channels, classes)

        # 2. instantiate the optimizer
        optimizer = optimizerClass()

        # 3. instantiate the train wrapper
        trainStrategy = train.ImageClassificationTrainStrategy(
            graph, model, optimizer)

        init = tf.global_variables_initializer()
        tf.add_to_collection("init", init.name)

    return trainStrategy


class BaseImageClassificationTest(unittest.TestCase):
    pass


def MNIST_must_converge(modelClass, optimizerClass,
                 epochs=20, batch_size=500):
    trainStrategy = generate_train_graph(
        modelClass, optimizerClass, 28, 28, 1, 10)

    train_writer = tf.summary.FileWriter("/tmp/%s/train" % "test")
    test_writer = tf.summary.FileWriter("/tmp/%s/test" % "test")

    print("logging at %s" % "/tmp/test//test")

    def train(epoch, dataset, batch_size, total, sess, summaries=True):
        average_loss = []
        average_error = []
        eye = np.eye(10)
        total_examples = 0
        error = 0

        def step_decay(epoch):
            initial_lrate = 0.1
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate 

        learning_rate = step_decay(epoch)
        #learning_rate = 0.01
        print(learning_rate)

        while total_examples != total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)
            # one hot encode 
            y_batch = eye[label_batch]
            opt = trainStrategy.optimize
            feed_dict = {
                trainStrategy.inputs: x_batch,
                trainStrategy.labels: y_batch,
                trainStrategy.learning_rate: learning_rate,
            }

            feed_dict.update(trainStrategy.train_parameters)
    
            fetches = [trainStrategy.optimize,
                       trainStrategy.loss,
                       trainStrategy.global_step,
                       trainStrategy.predictions,
                       trainStrategy.categorical_error,
                       ]

            _, loss, global_step, predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_loss.append(loss)
            average_error.append(error)

        if summaries:
            fetches = trainStrategy.summary_op
            summary = sess.run(fetches, feed_dict=feed_dict)
            train_writer.add_summary(summary)

        return global_step, np.mean(average_loss), np.mean(average_error) * 100.

    def test(epoch, dataset, batch_size, total, sess, summaries=True):
        total_examples = 0
        error = 0
        average_error = []
        eye = np.eye(10)
        while total_examples != total:
            x_batch, label_batch = dataset.next_batch(batch_size)
            total_examples += len(x_batch)

            feed_dict = {
                trainStrategy.inputs: x_batch,
                trainStrategy.labels: eye[label_batch],
            }

            fetches = [
                    trainStrategy.predictions,
                    trainStrategy.categorical_error,
                    ]

            predictions, error = sess.run(
                fetches, feed_dict=feed_dict)

            average_error.append(error)

        # Add summaries
        if summaries:
            fetches = trainStrategy.summary_op
            summary = sess.run(fetches, feed_dict=feed_dict)
            train_writer.add_summary(summary)

        return np.mean(average_error) * 100.0

    with tf.Session(graph=trainStrategy.graph) as sess:
        tf.set_random_seed(12345678)
        sess.run(tf.get_collection('init')[0])

        # from tensorflow.python import debug as tf_debug                                                                                                                                                                                                                                
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)    
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)       
        summaries = False

        dataset = read_data_sets('/tmp/deepwater/datasets/', validation_size=0)

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

        test_writer.close()
        train_writer.close()


