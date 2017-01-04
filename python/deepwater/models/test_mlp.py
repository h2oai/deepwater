import unittest

import numpy as np
import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


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
                 epochs=20, batch_size=128):
    trainStrategy = generate_train_graph(
        modelClass, optimizerClass, 28, 28, 1, 10)

    def train(epoch, dataset, batch_size, sess):
        average_loss = []
        average_error = []
        eye = np.eye(10)
        while epoch == dataset.epochs_completed:
            x_batch, label_batch = dataset.next_batch(batch_size)
            # one hot encode 
            y_batch = eye[label_batch]
            opt = trainStrategy.optimize
            feed_dict = {
                trainStrategy.inputs: x_batch,
                trainStrategy.labels: y_batch,
            }

            feed_dict.update(trainStrategy.train_parameters)

            fetches = [trainStrategy.optimize,
                       trainStrategy.loss,
                       trainStrategy.global_step,
                       trainStrategy.predictions,
                       ]

            _, loss, global_step, predictions = sess.run(
                fetches, feed_dict=feed_dict)
            error = 0. 
            for a,b in zip([p.argmax() for p in predictions[:]], label_batch):
                if a != b:
                   error += 1 
                else:
                   error = 0.
            average_loss.append(loss)
            average_error.append(error)
        return np.mean(average_loss), np.mean(average_error)

    def test(epoch, dataset, batch_size, sess):
        average_error = []
        while epoch == dataset.epochs_completed:
            x_batch, label_batch = dataset.next_batch(batch_size)

            feed_dict = {
                trainStrategy.inputs: x_batch,
            }

            fetches = [trainStrategy.predictions,]

            predictions = sess.run(
                fetches, feed_dict=feed_dict)

            error = 0. 
            for a,b in zip([p.argmax() for p in predictions[:]], label_batch):
                if a != b:
                   error += 1.0
                else:
                   error = 0.
            average_error.append(error)
        return np.mean(average_error)

    with tf.Session(graph=trainStrategy.graph) as sess:
        sess.run(tf.get_collection('init')[0])
        dataset = read_data_sets('/tmp/deepwater/datasets/')
        for epoch in range(epochs):
            train_loss, train_error = train(epoch, dataset.train, batch_size, sess)
            test_error = test(epoch, dataset.test, batch_size, sess)
            print('epoch:', epoch, 'train loss:', train_loss, 
                    'train error:', train_error,
                    'test error:', test_error)

        epoch = 0
        validation_err = test(epoch, dataset.validation, batch_size, sess)
        print('validation error:', validation_err)

from functools import partial

class TestMLP(unittest.TestCase):

    def test_single_layer(self):
        hidden_layers = [100]
        model = mlp.MultiLayerPerceptron

        MNIST_must_converge(model, optimizers.DefaultOptimizer, epochs=10)

    def XXtest_single_layer_with_dropout(self):
        hidden_layers = [1024, 1024]
        dropout = [0.8, 0.5, 0.5, ]
        dropout = []
        model = partial(mlp.MultiLayerPerceptron, hidden_layers=hidden_layers,
                dropout=dropout)

        MNIST_must_converge(model, optimizers.DefaultOptimizer, epochs=20)

if __name__ == "__main__":
    unittest.main()
