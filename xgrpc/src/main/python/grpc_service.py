from concurrent import futures

import sys
import os
import time
import json
import grpc
import traceback
import hashlib
import uuid
import tempfile

import numpy as np

import tensorflow as tf
import grpc_service_pb2 as pb

import logging
logger = logging.getLogger('tcpserver')

# from google.protobuf import text_format
from tensorflow.core.protobuf import meta_graph_pb2

from utils import *

tf.logging.set_verbosity(tf.logging.INFO)


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def deepwater_image_classification_model(
    x, y, logits, loss, accuracy, optimizer):
    # This is required by the h2o tensorflow backend

    # train
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))

    train_op = optimizer.apply_gradients(grads_and_vars=grads)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", accuracy)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)

    for grad, var in grads:
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + var.name + '/gradient', mean)
        tf.histogram_summary(var.name + '/gradient', grad)

    summary_op = tf.merge_all_summaries()

    tf.add_to_collection("train", train_op)
    tf.add_to_collection("summary", summary_op)

    tf.add_to_collection("logits", y)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    tf.add_to_collection("init", init.name)

    meta = json.dumps({
        "inputs": {"batch_image_input": x.name, "categorical_labels": y.name},
        "outputs": {"categorical_logits": logits.name},
        "metrics": {"accuracy": accuracy.name, "total_loss": loss.name},
        "parameters": {"global_step": global_step.name},
    })
    tf.add_to_collection("meta", meta)

    return tf.train.export_meta_graph(saver_def=saver.as_saver_def())


def _mlp(width=28, height=28, channels=1, classes=10):
    graph = tf.Graph()
    with graph.as_default():
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size], name="x")
        initialization = tf.truncated_normal(
            [size, classes], mean=0.2, stddev=0.)

        w = tf.Variable(initialization, name="W")
        b = tf.Variable(tf.zeros([classes]), name="b")

        variable_summaries(w, "W")
        variable_summaries(w, "b")

        y = tf.matmul(x, w) + b

        y = tf.nn.relu(y)

        # labels
        y_ = tf.placeholder(tf.float32, [None, classes], name="y")

        # accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1),
                                      tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # train
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_))
        optimizer = tf.train.GradientDescentOptimizer(0.01)

        return deepwater_image_classification_model(x, y_, y, loss, accuracy,
                optimizer)
# LeNet


def _lenet(width=28, height=28, channels=1, classes=10,
           clip_gradient=6, mini_batch_size=32):
    graph = tf.Graph()
    with graph.as_default():
        size = width * height * channels
        x = tf.placeholder(tf.float32, [None, size], name="x")

        # https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, mean=0.1, stddev=0.1)
            var = tf.Variable(initial, name=name)
            variable_summaries(var, name)
            return var

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            var = tf.Variable(initial)
            variable_summaries(var, name)
            return var

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        W_conv1 = weight_variable([5, 5, 1, 32], 'conv1/W')
        b_conv1 = bias_variable([32], 'conv1/bias')

        x_image = tf.reshape(x, [-1, width, height, channels])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64], 'conv2/W')
        b_conv2 = bias_variable([64], 'conv2/bias')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024], 'conv3/W')
        b_fc1 = bias_variable([1024], 'conv3/bias')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder_with_default(1.0, [], name="dropout")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, classes], 'fc1/W')
        b_fc2 = bias_variable([classes], 'fc1/bias')

        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # labels
        y_= tf.placeholder(tf.float32, [None, classes], name = "y")

        # accuracy
        correct_prediction=tf.equal(tf.argmax(y, 1),
                                      tf.argmax(y_, 1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_))

        optimizer = tf.train.AdamOptimizer(1e-4)
        # optimizer = tf.train.GradientDescentOptimizer(0.01)

        return deepwater_image_classification_model(x, y_, y, 
                loss, 
                accuracy,
                optimizer) 


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
        self._summaries_dir = os.path.join(tempfile.mkdtemp(), self.handle)

    def __init_session_if_required(self, meta_graph_def):
        if self._session:
            return
        print("initializing session %s" % self.handle)
        graph = tf.Graph()
        with graph.as_default():
            self._session = tf.Session(graph=graph)

            self._saver = tf.train.import_meta_graph(meta_graph_def)

            init = graph.get_collection('init')[0]
            meta = graph.get_collection('meta')[0]
            summary_op = graph.get_collection('summary')[0]

            train_op = graph.get_collection('train')[0]
            meta = json.loads(meta.decode('utf8'))
            for k, v in list(meta.items()):
                meta.update(v)

            self._session.run(init)

            self._meta = meta
            self._meta['train'] = train_op
            self._meta['summary'] = summary_op 
            self._reverse_meta = {}
            for k, v in meta.items():
                if isinstance(v, str):
                    self._reverse_meta[v] = k

            print("summaries: %s" % self._summaries_dir)
            self._train_writer = tf.train.SummaryWriter(self._summaries_dir,
                                                          self._session.graph)

    def update_fetches(self, fetches):
        return [self._meta[k] if self._meta.get(k) else k for k in fetches]

    def update_feeds(self, feeds):
        return dict(((self._meta[k], v)
                     if self._meta.get(k) else(k, v))
                    for (k, v) in feeds.items())

    def run(self, model, fetches, feeds):
        self.__init_session_if_required(model)
        fetches = self.update_fetches(fetches)
        feeds = self.update_feeds(feeds)
        self._global_step += 1
        feeds[self._meta['global_step']] = self._global_step

        if self._global_step % 10 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            fetches.append(self._meta['summary'])
            result = self._session.run(fetches,
                              feed_dict=feeds,
                              options=run_options,
                              run_metadata=run_metadata)
            summary = result.pop()
            self._train_writer.add_run_metadata(run_metadata, 
                        'step%d' % self._global_step
                    )
            self._train_writer.add_summary(summary, self._global_step)
            return result

        return self._session.run(fetches,
                feed_dict=feeds)

    def finalize(self):
        if self._session:
            self._session.close()
            print("finalizing session %s" % self.handle)

    def save_model_variables(self, model, path=None):
        self.__init_session_if_required(model)
        sdef = self._saver.saver_def
        feeds = {
            sdef.filename_tensor_name: path,
        }
        fetches = [sdef.save_tensor_name]
        self._session.run(fetches, feeds)

    def load_model_variables(self, model, path=None):
        self.__init_session_if_required(model)
        sdef = self._saver.saver_def
        feeds = {
            sdef.filename_tensor_name: path,
        }
        fetches = [sdef.save_tensor_name]
        self._session.run(fetches, feeds)

    def load_model(self, path=None):
        self.__init_session_if_required(path)

    def save_model(self, model, path=None):
        self.__init_session_if_required(model)
        tf.train.export_meta_graph(filename=path,
                        saver_def=self._saver.as_saver_def(),
                        graph=self._session.graph)


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
            options = convert_map(req.params)
            session.save_model(model, **options)
            return pb.SaveModelResponse(
                status=pb.Status(ok=True)
            )
        except Exception as e:
            traceback.print_exc()
            return pb.SaveModelResponse(
                status=pb.Status(ok=False, message=str(e))
            )

    def LoadModel(self, req, ctx):
        try:
            session = self._get_session(req)
            options = convert_map(req.params)
            model = session.load_model(**options)
            return pb.LoadModelResponse(
                model=model,
                status=pb.Status(ok=True)
            )
        except Exception as e:
            traceback.print_exc()
            return pb.LoadModelResponse(
                status=pb.Status(ok=False, message=str(e))
            )

    def LoadModelVariables(self, req, ctx):
        try:
            session = self._get_session(req)
            model = self._get_model(req)
            options = convert_map(req.params)
            session.load_model_variables(model, **options)
            return pb.LoadModelVariablesResponse(status=pb.Status(ok=True))
        except Exception as e:
            traceback.print_exc()
            return pb.LoadModelVariablesResponse(
                status=pb.Status(ok=False, message=str(e))
            )

    def SaveModelVariables(self, req, ctx):
        try:
            session = self._get_session(req)
            model = self._get_model(req)
            options = convert_map(req.params)
            session.save_model_variables(model, **options)
            return pb.SaveModelVariablesResponse(status=pb.Status(ok=True))
        except Exception as e:
            traceback.print_exc()
            return pb.SaveModelVariablesResponse(
                status=pb.Status(ok=False, message=str(e))
            )

    def SetModelParameters(self, req, ctx):
        try:
            session = self._get_session(req)
            model = self._get_model(req)
            options = convert_map(req.params)
            session.set_model_params(model, **options)
            return pb.SetModelParametersResponse(status=pb.Status(ok=True))
        except Exception as e:
            traceback.print_exc()
            return pb.SetModelParametersResponse(
                status=pb.Status(ok=False, message=str(e))
            )

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
            return pb.DeleteSessionResponse(status=pb.Status(ok=True))
        except Exception as e:
            traceback.print_exc()
            return pb.DeleteSessionResponse(
                status=pb.Status(ok=False, message=str(e))
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
