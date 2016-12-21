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


class DeepWaterPredictBackendStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Execute = channel.unary_unary(
        '/deepwater.DeepWaterPredictBackend/Execute',
        request_serializer=grpc__service__pb2.ExecuteRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.ExecuteResponse.FromString,
        )


class DeepWaterPredictBackendServicer(object):

  def Execute(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DeepWaterPredictBackendServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Execute': grpc.unary_unary_rpc_method_handler(
          servicer.Execute,
          request_deserializer=grpc__service__pb2.ExecuteRequest.FromString,
          response_serializer=grpc__service__pb2.ExecuteResponse.SerializeToString,
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
        response_deserializer=grpc__service__pb2.DeleteSessionResponse.FromString,
        )
    self.LoadModel = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/LoadModel',
        request_serializer=grpc__service__pb2.LoadModelRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.LoadModelResponse.FromString,
        )
    self.SaveModel = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SaveModel',
        request_serializer=grpc__service__pb2.SaveModelRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.SaveModelResponse.FromString,
        )
    self.LoadModelWeights = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/LoadModelWeights',
        request_serializer=grpc__service__pb2.LoadModelWeightsRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.LoadModelWeightsResponse.FromString,
        )
    self.SaveModelWeights = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SaveModelWeights',
        request_serializer=grpc__service__pb2.SaveModelWeightsRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.SaveModelWeightsResponse.FromString,
        )
    self.SetModelParameters = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/SetModelParameters',
        request_serializer=grpc__service__pb2.SetModelParametersRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.SetModelParametersResponse.FromString,
        )
    self.Execute = channel.unary_unary(
        '/deepwater.DeepWaterTrainBackend/Execute',
        request_serializer=grpc__service__pb2.ExecuteRequest.SerializeToString,
        response_deserializer=grpc__service__pb2.ExecuteResponse.FromString,
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

  def LoadModelWeights(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SaveModelWeights(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SetModelParameters(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Execute(self, request, context):
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
          response_serializer=grpc__service__pb2.DeleteSessionResponse.SerializeToString,
      ),
      'LoadModel': grpc.unary_unary_rpc_method_handler(
          servicer.LoadModel,
          request_deserializer=grpc__service__pb2.LoadModelRequest.FromString,
          response_serializer=grpc__service__pb2.LoadModelResponse.SerializeToString,
      ),
      'SaveModel': grpc.unary_unary_rpc_method_handler(
          servicer.SaveModel,
          request_deserializer=grpc__service__pb2.SaveModelRequest.FromString,
          response_serializer=grpc__service__pb2.SaveModelResponse.SerializeToString,
      ),
      'LoadModelWeights': grpc.unary_unary_rpc_method_handler(
          servicer.LoadModelWeights,
          request_deserializer=grpc__service__pb2.LoadModelWeightsRequest.FromString,
          response_serializer=grpc__service__pb2.LoadModelWeightsResponse.SerializeToString,
      ),
      'SaveModelWeights': grpc.unary_unary_rpc_method_handler(
          servicer.SaveModelWeights,
          request_deserializer=grpc__service__pb2.SaveModelWeightsRequest.FromString,
          response_serializer=grpc__service__pb2.SaveModelWeightsResponse.SerializeToString,
      ),
      'SetModelParameters': grpc.unary_unary_rpc_method_handler(
          servicer.SetModelParameters,
          request_deserializer=grpc__service__pb2.SetModelParametersRequest.FromString,
          response_serializer=grpc__service__pb2.SetModelParametersResponse.SerializeToString,
      ),
      'Execute': grpc.unary_unary_rpc_method_handler(
          servicer.Execute,
          request_deserializer=grpc__service__pb2.ExecuteRequest.FromString,
          response_serializer=grpc__service__pb2.ExecuteResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'deepwater.DeepWaterTrainBackend', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
