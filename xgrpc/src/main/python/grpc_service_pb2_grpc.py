import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2
import grpc_service_pb2 as grpc__service__pb2


class DeepWaterPredictBackendStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/deepwater.DeepWaterPredictBackend/Predict',
        request_serializer=grpc__service__pb2.PredictRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )


class DeepWaterPredictBackendServicer(object):

  def Predict(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DeepWaterPredictBackendServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=grpc__service__pb2.PredictRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'deepwater.DeepWaterPredictBackend', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class DeepWaterTrainBackendStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.CreateModel = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/CreateModel',
        request_serializer=grpc__service__pb2.CreateModelRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.CreateModelResponse.FromString,
        )
    self.CreateSession = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/CreateSession',
        request_serializer=grpc__service__pb2.CreateSessionRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.CreateSessionResponse.FromString,
        )
    self.DeleteSession = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/DeleteSession',
        request_serializer=grpc__service__pb2.DeleteSessionRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )
    self.LoadModel = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/LoadModel',
        request_serializer=grpc__service__pb2.LoadModelRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )
    self.SaveModel = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SaveModel',
        request_serializer=grpc__service__pb2.SaveModelRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )
    self.LoadWeights = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/LoadWeights',
        request_serializer=grpc__service__pb2.LoadWeightsRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )
    self.SaveWeights = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SaveWeights',
        request_serializer=grpc__service__pb2.SaveWeightsRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.Status.FromString,
        )
    self.SetParameters = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SetParameters',
        request_serializer=grpc__service__pb2.SetParametersRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.SetParametersResponse.FromString,
        )
    self.Train = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/Train',
        request_serializer=grpc__service__pb2.TrainRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.TrainResponse.FromString,
        )
    self.Predict = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/Predict',
        request_serializer=grpc__service__pb2.PredictRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.PredictResponse.FromString,
        )


class DeepWaterTrainBackendServicer(object):

  def CreateModel(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreateSession(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DeleteSession(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadModel(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SaveModel(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadWeights(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SaveWeights(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SetParameters(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Train(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Predict(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DeepWaterTrainBackendServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'CreateModel': grpc.unary_unary_rpc_method_handler(
          servicer.CreateModel,
          request_deserializer=grpc__service__pb2.CreateModelRequest.FromString,
          response_serializer=grpc__service__pb2.CreateModelResponse.SerializeToString,
      ),
      'CreateSession': grpc.unary_unary_rpc_method_handler(
          servicer.CreateSession,
          request_deserializer=grpc__service__pb2.CreateSessionRequest.FromString,
          response_serializer=grpc__service__pb2.CreateSessionResponse.SerializeToString,
      ),
      'DeleteSession': grpc.unary_unary_rpc_method_handler(
          servicer.DeleteSession,
          request_deserializer=grpc__service__pb2.DeleteSessionRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
      'LoadModel': grpc.unary_unary_rpc_method_handler(
          servicer.LoadModel,
          request_deserializer=grpc__service__pb2.LoadModelRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
      'SaveModel': grpc.unary_unary_rpc_method_handler(
          servicer.SaveModel,
          request_deserializer=grpc__service__pb2.SaveModelRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
      'LoadWeights': grpc.unary_unary_rpc_method_handler(
          servicer.LoadWeights,
          request_deserializer=grpc__service__pb2.LoadWeightsRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
      'SaveWeights': grpc.unary_unary_rpc_method_handler(
          servicer.SaveWeights,
          request_deserializer=grpc__service__pb2.SaveWeightsRequest.FromString,
          response_serializer=grpc__service__pb2.Status.SerializeToString,
      ),
      'SetParameters': grpc.unary_unary_rpc_method_handler(
          servicer.SetParameters,
          request_deserializer=grpc__service__pb2.SetParametersRequest.FromString,
          response_serializer=grpc__service__pb2.SetParametersResponse.SerializeToString,
      ),
      'Train': grpc.unary_unary_rpc_method_handler(
          servicer.Train,
          request_deserializer=grpc__service__pb2.TrainRequest.FromString,
          response_serializer=grpc__service__pb2.TrainResponse.SerializeToString,
      ),
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=grpc__service__pb2.PredictRequest.FromString,
          response_serializer=grpc__service__pb2.PredictResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'deepwater.DeepWaterTrainBackend', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
