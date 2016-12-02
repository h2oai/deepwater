from concurrent import futures
import time

import json

import grpc

import service_pb2
import tensorflow as tf


def convert(req):
    result = {}
    for key, param in req.params.iteritems():
        field = param.WhichOneof('value')
        value = getattr(param, field)
        result[key] = value
    return result

def mlp(width=28, height=28, channels=1, classes=10):
    graph = tf.Graph()
    with graph.as_default():
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size])
        W = tf.Variable(tf.zeros([size, classes]))
        b = tf.Variable(tf.zeros([classes]))
        y = tf.matmul(x, W) + b

        # labels
        y_ = tf.placeholder(tf.float32, [None, classes])

        # accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1),
                                       tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # train
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        tf.add_to_collection("train", train_step)
        # this is required by the h2o tensorflow backend
        global_step = tf.Variable(0, name="global_step", trainable=False)

        init = tf.initialize_all_variables()
        tf.add_to_collection("init", init)
        tf.add_to_collection("logits", y)
        saver = tf.train.Saver()
        meta = json.dumps({
                "inputs": {"batch_image_input": x.name, "categorical_labels": y_.name},
                "outputs": {"categorical_logits": y.name},
                "metrics": {"accuracy": accuracy.name, "total_loss": cross_entropy.name},
                "parameters": {"global_step": global_step.name},
        })
        tf.add_to_collection("meta", meta)
        filename = "/tmp/lenet_tensorflow.meta"
        return tf.train.export_meta_graph(filename, saver_def=saver.as_saver_def())


class DeepWaterServer(service_pb2.ServiceServicer):

  """Provides methods that implement functionality of DeepWater server."""

  def Ping(self, *args):
    return service_pb2.Status()

  def BuildNetwork(self, req, ctx):
    params = convert(req)
    name = params.pop('name')
    print(name, params)
    return service_pb2.NetworkResponse(
        network=mlp(**params),
        status=service_pb2.Status(),
    )


def serve():
  port = 50051
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service_pb2.add_ServiceServicer_to_server(DeepWaterServer(), server)
  server.add_insecure_port('[::]:%d' % port)
  server.start()
  print("listening on port {}".format(port))
  try:
    while True:
      time.sleep(10)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  serve()
