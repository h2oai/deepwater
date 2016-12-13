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
    w = tf.Variable(tf.zeros([size, classes]), name="W")
    b = tf.Variable(tf.zeros([classes]), name="b")
    y = tf.matmul(x, w) + b

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
    filename = "/tmp/xxxlenet_tensorflow.meta"
    return tf.train.export_meta_graph(saver_def=saver.as_saver_def())

def convert_fetches(params):
    return [ p.name for p in params]

def convert_feeds(params):
    feeds = {}
    for p in params:
      assert isinstance(p, pb.Tensor)
      # FIXME
      a = np.array(p.float_value)
      # print(p.name)
      # print(a.shape)
      a = a.reshape((10, -1))
      feeds[p.name] = a #np.array(p.float_value).reshape((-1, 10))

    feeds['global_step'] = 0
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

        print(meta)


  def update_fetches(self, fetches):
    return [ self._meta[k] if self._meta.get(k) else k for k in fetches ]

  def update_feeds(self, feeds):
    return dict( ((self._meta[k], v)
                  if self._meta.get(k) else(k, v))
                  for (k,v) in feeds.items())

  def train(self, model, fetches, feeds):
    self._init_session_if_required(model)
    fetches = self.update_fetches(fetches)
    feeds = self.update_feeds(feeds)
    print("train", fetches)
    return self._session.run(fetches, feed_dict=feeds)

  def predict(self, model, fetches, feeds):
    print("predict", fetches)
    self._init_session_if_required(model)
    fetches = self.update_fetches(fetches)
    feeds = self.update_feeds(feeds)
    return self._session.run(fetches, feed_dict=feeds)

  def finalize(self):
    if self._session:
        self._session.close()


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
          session = pb.Session(
             handle = s.handle,
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

      #serialized = model.SerializeToString()
      checksum = _checksum(serialized)

      return pb.CreateModelResponse(
          model = pb.BackendModel(
             id = checksum,
             state = serialized
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
      session.save(model)
      return pb.Status(ok=True)
    except Exception as e:
      traceback.print_exc()
      return pb.Status(ok=False, message=str(e))

  def LoadModel(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.load(model)
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

  def Train(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      session.train(model,
                convert_fetches(req.fetches),
                convert_feeds(req.feeds))
      return pb.TrainResponse(
          status=pb.Status(ok=True),
      )
    except Exception as e:
      traceback.print_exc()
      return pb.TrainResponse(
          status=pb.Status(ok=False, message=str(e)),
      )

  def Predict(self, req, ctx):
    try:
      session = self._get_session(req)
      model = self._get_model(req)
      result = session.predict(model,
                            convert_fetches(req.fetches),
                            convert_feeds(req.feeds))


      return pb.PredictResponse(
          status=pb.Status(ok=True),
      )
    except Exception as e:
      traceback.print_exc()
      return pb.PredictResponse(
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
      return pb.PredictResponse(
          status=pb.Status(ok=False, message=str(e)),
      )



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
