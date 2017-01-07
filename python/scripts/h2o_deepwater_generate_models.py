from __future__ import division
from __future__ import print_function

import locale
import json

from six import PY2, PY3
from six.moves import cPickle
from six.moves import range

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops

from google.protobuf import text_format

from deepwater.models import (lenet, mlp, alexnet)
from deepwater import train, optimizers

from functools import partial

tf.logging.set_verbosity(tf.logging.INFO)


def test_cifar10(modelClass, optimizerClass, 
                 epochs=20, batch_size=128):
    data = {'data': [], 'labels': []}
    for batch in range(1, 6):
        filename = '/datasets/cifar-10-batches-py/data_batch_%d' % batch
        with open(filename, 'rb') as fd:
            if PY2:
                d = cPickle.load(fd)
            else:
                d = cPickle.load(fd, encoding='latin1')
            data['data'].extend(d['data'])
            data['labels'].extend(d['labels'])

    trainStrategy = generate_train_graph(
        modelClass, optimizerClass, 32, 32, 3, 10)

    with tf.Session(graph=trainStrategy.graph) as sess:
        sess.run(tf.get_collection('init')[0])
        for epoch in range(epochs):
            x_batch = []
            y_batch = []
            for image, label in zip(data['data'], data['labels']):
                image = np.array(image, dtype=np.float32).reshape(32 * 32 * 3)
                x_batch.append(image)
                one_hot = np.zeros([10], dtype=np.float32)
                one_hot[label] = 1
                y_batch.append(one_hot)

                if len(y_batch) == batch_size:
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
                    global_step += 1

                    print(epoch, global_step, loss)
                    x_batch = []
                    y_batch = []


def print_stat(prefix, statistic_type, value):
    if value is None:
        friendly_value = "None"
    else:
        friendly_value = locale.format("%d", value, grouping=True)
    print("%s%s=%s" % (prefix, statistic_type, friendly_value))


def calculate_graph_metrics(graph_def, statistic_types, input_layer,
                            input_shape_override, batch_size):
    """Looks at the performance statistics of all nodes in the graph."""
    _ = tf.import_graph_def(graph_def, name="")
    total_stats = {}
    node_stats = {}
    for statistic_type in statistic_types:
        total_stats[statistic_type] = ops.OpStats(statistic_type)
        node_stats[statistic_type] = {}
    # Make sure we get pretty-printed numbers with separators.
    locale.setlocale(locale.LC_ALL, "")
    with tf.Session() as sess:
        input_tensor = sess.graph.get_tensor_by_name(input_layer)
        input_shape_tensor = input_tensor.get_shape()
        if input_shape_tensor:
            input_shape = input_shape_tensor.as_list()
        else:
            input_shape = None
        if input_shape_override:
            input_shape = input_shape_override
        if input_shape is None:
            raise ValueError("""No input shape was provided on the command line,"""
                             """ and the input op itself had no default shape, so"""
                             """ shape inference couldn't be performed. This is"""
                             """ required for metrics calculations.""")
        input_shape[0] = batch_size
        input_tensor.set_shape(input_shape)
        for node in graph_def.node:
            # Ensure that the updated input shape has been fully-propagated before we
            # ask for the statistics, since they may depend on the output size.
            op = sess.graph.get_operation_by_name(node.name)
            ops.set_shapes_for_outputs(op)
            for statistic_type in statistic_types:
                current_stats = ops.get_stats_for_node_def(sess.graph, node,
                                                           statistic_type)
                node_stats[statistic_type][node.name] = current_stats
                total_stats[statistic_type] += current_stats
    return total_stats, node_stats


def generate_models(name, modelClass):
    height = [28, 32, 224, 320]
    width = [28, 32, 224, 320]
    channels = [1, 3]
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 1000]

    for (h, w) in zip(height, width):
        for ch in channels:
            for class_n in classes:
                filename = "%s_%dx%dx%d_%d" % (name, h, w, ch, class_n)
                model = modelClass([h, w, ch], [class_n])
                model.export(filename + ".meta")


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


def export_train_graph(modelClass, optimizerClass,
                       width, height, channels, classes):
    graph = tf.Graph()
    with graph.as_default():
        # 1. instantiate the model
        model = modelClass(width, height, channels, classes)

        # 2. instantiate the optimizer
        optimizer = optimizerClass()

        # 3. instantiate the train wrapper
        trainStrategy = train.ImageClassificationTrainStrategy(
            graph,
            model, optimizer)

        # 4. export train graph
        filename = "%s_%dx%dx%d_%d.pb" % (model.name, 
                height, 
                width, 
                channels,
                classes)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf.add_to_collection("init", init.name)

        meta = json.dumps({
            "inputs": {"batch_image_input": trainStrategy.inputs.name,
                       "categorical_labels": trainStrategy.labels.name},
            "outputs": {"categorical_logits": model.logits.name},
            "metrics": {"accuracy": trainStrategy.accuracy.name,
                        "total_loss": trainStrategy.loss.name},
            "parameters": {"global_step": trainStrategy.global_step.name},
        })

        tf.add_to_collection("meta", meta)

        tf.train.export_meta_graph(filename=filename, 
                                   saver_def=saver.as_saver_def())
        print("model exported to ", filename)

def export_model_graph(modelClass):
    height = [28, 32, 224]
    width = [28, 32, 224]
    channels = [1, 3]
    classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 1000]
    for (h, w) in zip(height, width):
        for ch in channels:
            for class_n in classes:
                m = export_train_graph(modelClass,
                        optimizers.MomentumOptimizer, h, w, ch, class_n)


if __name__ == "__main__":
    # generate MLP
    default_mlp = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=[2048, 2048, 2048], 
                        dropout=[0.2, 0.5, 0.5]) 

    gen = export_model_graph(default_mlp)

    for model in ( lenet.LeNet, alexnet.AlexNet ):
        export_model_graph(model)
