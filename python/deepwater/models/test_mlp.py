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


def MNIST_must_converge(modelClass, optimizerClass,
                 epochs=20, batch_size=128):
    trainStrategy = generate_train_graph(
        modelClass, optimizerClass, 28, 28, 1, 10)

    with tf.Session(graph=trainStrategy.graph) as sess:
        sess.run(tf.get_collection('init')[0])
        dataset = read_data_sets('/tmp/deepwater/datasets/')
        for epoch in range(epochs):
            while epoch == dataset.train.epochs_completed:
                x_batch, y_batch = dataset.train.next_batch(batch_size)
                # one hot encode 
                y_batch = np.eye(10)[y_batch]
                opt = trainStrategy.optimize
                feed_dict = {
                    trainStrategy.inputs: x_batch,
                    trainStrategy.labels: y_batch,
                }
                fetches = [trainStrategy.optimize,
                           trainStrategy.loss,
                           trainStrategy.global_step
                           ]

                _, loss, global_step = sess.run(
                    fetches, feed_dict=feed_dict)

                print(epoch, global_step, loss)

            while epoch == dataset.test.epochs_completed:
                x_batch, y_batch = dataset.test.next_batch(batch_size)
                # one hot encode 
                y_batch = np.eye(10)[y_batch]
                opt = trainStrategy.optimize
                feed_dict = {
                    trainStrategy.inputs: x_batch,
                    trainStrategy.labels: y_batch,
                }
                fetches = [
                           trainStrategy.loss,
                           trainStrategy.global_step
                           ]

                _, loss, global_step = sess.run(
                    fetches, feed_dict=feed_dict)

                print("test:", epoch, global_step, loss)

        epoch = 0
        while epoch == dataset.validation.epochs_completed:
            x_batch, y_batch = dataset.validation.next_batch(batch_size)
            # one hot encode 
            y_batch = np.eye(10)[y_batch]
            opt = trainStrategy.optimize
            feed_dict = {
                trainStrategy.inputs: x_batch,
                trainStrategy.labels: y_batch,
            }
            fetches = [
                       trainStrategy.loss,
                       trainStrategy.global_step
                       ]

            _, loss, global_step = sess.run(
                fetches, feed_dict=feed_dict)

            print("validation:", epoch, global_step, loss)


from functools import partial

class TestMLP(unittest.TestCase):

    def _test_single_layer(self):
        hidden_layers = [100]
        dropout = [1.0]
        model = mlp.MultiLayerPerceptron
        # (
        #         hidden_layers=hidden_layers, 
        #         dropout=dropout)

        MNIST_must_converge(model, optimizers.DefaultOptimizer, epochs=10)

    def test_single_layer_with_dropout(self):
        hidden_layers = [100]
        dropout = [0.5]
        model = partial(mlp.MultiLayerPerceptron, hidden_layers=hidden_layers,
                dropout=dropout)

        MNIST_must_converge(model, optimizers.DefaultOptimizer, epochs=10)

if __name__ == "__main__":
    unittest.main()
