from concurrent import futures

import sys
import os
import time
import json
import grpc
import traceback
import hashlib
import uuid

import numpy as np

import tensorflow as tf
import grpc_service_pb2 as pb

import logging
logger = logging.getLogger('tcpserver')

# from google.protobuf import text_format
from tensorflow.core.protobuf import meta_graph_pb2


def _mlp(width=28, height=28, channels=1, classes=10):
  graph = tf.Graph()
  with graph.as_default():
    size = width * height * channels
    x = tf.placeholder(tf.float32, [None, size], name="x")
    initialization = tf.truncated_normal([size, classes], mean=0.5, stddev=0.2)
    w = tf.Variable(initialization, name="W")
    b = tf.Variable(tf.zeros([classes]), name="b")
    y = tf.matmul(x, w) + b

    y = tf.nn.relu(y)

    # labels
    y_ = tf.placeholder(tf.float32, [None, classes], name="y")

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    tf.add_to_collection("train", train_step)
    # this is required by the h2o tensorflow backend
    global_step = tf.Variable(0, name="global_step", trainable=False)

    tf.add_to_collection("logits", y)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    tf.add_to_collection("init", init.name)

    meta = json.dumps({
        "inputs": {"batch_image_input": x.name, "categorical_labels": y_.name},
        "outputs": {"categorical_logits": y.name},
        "metrics": {"accuracy": accuracy.name, "total_loss": cross_entropy.name},
        "parameters": {"global_step": global_step.name},
    })
    tf.add_to_collection("meta", meta)

    # FIXME
    return tf.train.export_meta_graph(saver_def=saver.as_saver_def())

# LeNet

def _lenet(width=28, height=28, channels=1, classes=10, clip_gradient=6, mini_batch_size=32):
  graph = tf.Graph()
  with graph.as_default():
    size = width * height * channels
    x = tf.placeholder(tf.float32, [None, size], name="x")

    # #---
    # initialization = tf.truncated_normal([size, classes], mean=0.5, stddev=0.2)
    # w = tf.Variable(initialization, name="W")
    # b = tf.Variable(tf.zeros([classes]), name="b")
    # y = tf.matmul(x, w) + b
    #
    # y = tf.nn.relu(y)
    # #---

    # https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,width,height,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, classes])
    b_fc2 = bias_variable([classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = y_conv

    # labels
    y_ = tf.placeholder(tf.float32, [None, classes], name="y")

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    tf.add_to_collection("train", train_step)
    # this is required by the h2o tensorflow backend
    global_step = tf.Variable(0, name="global_step", trainable=False)

    tf.add_to_collection("logits", y)
    saver = tf.train.Saver()
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    tf.add_to_collection("init", init.name)

    meta = json.dumps({
      "inputs": {"batch_image_input": x.name, "categorical_labels": y_.name},
      "outputs": {"categorical_logits": y.name},
      "metrics": {"accuracy": accuracy.name, "total_loss": cross_entropy.name},
      "parameters": {"global_step": global_step.name},
    })
    tf.add_to_collection("meta", meta)

    # FIXME
    return tf.train.export_meta_graph(saver_def=saver.as_saver_def())


def convert_fetches(params):
  return [p.name for p in params]


def convert_feeds(params):
  feeds = {}
  for p in params:
    assert isinstance(p, pb.Tensor)
    # FIXME
    a = np.array(p.float_value)
    # print(p.name)
    # print(a.shape)
    a = a.reshape((10, -1))
    feeds[p.name] = a  # np.array(p.float_value).reshape((-1, 10))
  return feeds


def convert_map(params):
  "Convert from a pb request param to a python map"

  result = {}
  for key, param in params.items():
    field = param.WhichOneof('value')
    value = getattr(param, field)
    result[key] = value
  return result


def createModel(modelName, params):
  if modelName == "mlp":
    tf_model = _mlp(**params)
    serialized = tf_model.SerializeToString()
    return serialized
  elif modelName == "lenet":
    tf_model = _lenet(**params)
    serialized = tf_model.SerializeToString()
    return serialized
  raise Exception("unsupported model {}".format(modelName))


def _checksum(data):
  hashvalue = hashlib.new('sha')
  hashvalue.update(data)
  return hashvalue.digest()


class _Session(object):

  def __init__(self, **options):
    self.handle = uuid.uuid4().hex
    self.options = options
    self.loaded_models = []
    self._session = None
    self._global_step = 0

  def _init_session_if_required(self, meta_graph_def):
    if self._session:
      return
    graph = tf.Graph()
    with graph.as_default():
      self._session = tf.Session(graph=graph)

      model = tf.train.import_meta_graph(meta_graph_def)

      #model.restore(self._session, filename)

      meta = graph.get_collection('meta')[0]

      meta = json.loads(meta.decode('utf8'))
      for k, v in list(meta.items()):
        meta.update(v)

      self._session.run(tf.initialize_all_variables())

      self._meta = meta

      self._reverse_meta = {}
      for k, v in meta.items():
        if isinstance(v, str):
          self._reverse_meta[v] = k

  def update_fetches(self, fetches):
    return [self._meta[k] if self._meta.get(k) else k for k in fetches]

  def update_feeds(self, feeds):
    return dict(((self._meta[k], v)
                 if self._meta.get(k) else(k, v))
                for (k, v) in feeds.items())

  def run(self, model, fetches, feeds):
    self._init_session_if_required(model)
    fetches = self.update_fetches(fetches)
    feeds = self.update_feeds(feeds)
    self._global_step += 1
    feeds[self._meta['global_step']] = self._global_step

    # accuracy = self._session.run(self._meta['accuracy'], feed_dict=feeds)

    return self._session.run(fetches, feed_dict=feeds)

  def finalize(self):
    if self._session:
      self._session.close()

  def save_model(self, model, path):
    self.__init_session_if_required(model)
    return self._session.run(fetches, feed_dict=feeds)

  def save_weights(self, model, path):
    self.__init_session_if_required(model)
    return self._session.run(fetches, feed_dict=feeds)

  def load_weights(self, model, path):
    self.__init_session_if_required(model)
    feeds[self._meta['save_filename']] = path
    feeds[self._meta['global_step']] = self._global_step

    # accuracy = self._session.run(self._meta['accuracy'], feed_dict=feeds)

    return self._session.run(fetches, feed_dict=feeds)

  def save_model(self, model, path):
    self.__init_session_if_required(model)
    return self._session.run(fetches, feed_dict=feeds)


def convert_tensors(fetches, numpy_tensor_list):
  result = []
  for name, np_array in zip(fetches, numpy_tensor_list):
    proto_tensor = pb.Tensor(
        name=name,
        float_value=np_array.reshape((-1,)).tolist()
    )
    result.append(proto_tensor)
  return result


class DeepWaterServer(pb.DeepWaterTrainBackendServicer):

  """Provide DeepWaterGrpcService."""

  __session_cache = {}
  __model_cache = {}

  def _get_session(self, req):
    return self.__session_cache.get(req.session.handle)

  def _get_model(self, req):
    model = self.__model_cache.get(req.model.id)
    if model is None:
      meta_graph_def = meta_graph_pb2.MetaGraphDef()
      meta_graph_def.ParseFromString(req.model.state)
      self.__model_cache[req.model.id] = meta_graph_def
      model = meta_graph_def
    return model

  def CreateSession(self, req, ctx):
    try:
      s = _Session()
      self.__session_cache[s.handle] = s
      return pb.CreateSessionResponse(
          session=pb.Session(
              handle=s.handle,
          ),
          status=pb.Status(ok=True),
      )
    except Exception as e:
      traceback.print_exc()
      return pb.CreateSessionResponse(
          status=pb.Status(ok=False, message=str(e)),
      )

  def CreateModel(self, req, ctx):
    try:
      session = self._get_session(req)
      serialized = createModel(req.modelName, convert_map(req.params))

      checksum = _checksum(serialized)

      return pb.CreateModelResponse(
          model=pb.BackendModel(
              id=checksum,
              state=serialized
          ),
          status=pb.Status(ok=True),
      )
    except Exception as e:
      traceback.print_exc()
      return pb.CreateModelResponse(
          status=pb.Status(ok=False, message=str(e)),
      )

  def SaveModel(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.save_model(model, req.path)
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def LoadModel(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.load_model(model, req.path)
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def LoadWeights(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.load_weights(model, req.path)
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def SaveWeights(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.save_weights(model, req.path)
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def SetParameters(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.set_params(model, convert_map(req.params))
      return pb.SetParametersResponse(status=pb.Status(ok=True))
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def Execute(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      fetches_name_list = convert_fetches(req.fetches)
      results = session.run(model,
                            fetches_name_list,
                            convert_feeds(req.feeds))

      results = convert_tensors(fetches_name_list, results)

      response = pb.ExecuteResponse(
          fetches=results,
          status=pb.Status(ok=True),
      )
      return response
    except Exception as e:
      traceback.print_exc()
      return pb.ExecuteResponse(
          status=pb.Status(ok=False, message=str(e)),
      )

  def DeleteSession(self, req, ctx):
    try:
      session = self._get_session(req)
      session.finalize()
      del self.__session_cache[session.handle]
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))


def serve(infile, outfile):
  print("starting DeepWater python Service")
  port = os.getenv('DEEPWATER_WORKER_PORT', '50051')
  max_workers = os.getenv('DEEPWATER_MAX_WORKERS', '10')
  port = int(port)
  max_workers = int(max_workers)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
  pb.add_DeepWaterTrainBackendServicer_to_server(DeepWaterServer(), server)
  server.add_insecure_port('[::]:%d' % port)
  server.start()

  logging.debug("listening on port {}".format(port))
  try:
    while True:
      time.sleep(10)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  serve(sys.stdin, sys.stdout)
