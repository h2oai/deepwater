from __future__ import print_function

import six
import grpc

import grpc_service_pb2 as pb


def check(status):
  if status.ok: return
  raise Exception(status.message)


def convert_value(value):
    pv = pb.ParamValue()
    if isinstance(value, six.integer_types):
        pv.i = value
    elif isinstance(value, six.string_types):
        pv.s = value
    elif isinstance(value, float):
        pv.f = value
    else:
        raise ValueError("not supported type:"+type(value))
    return pv


def encode_params(options):
    encoded = {}
    for key, value in six.iteritems(options):
        encoded[key] = convert_value(value)
    return encoded

def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = pb.DeepWaterTrainBackendStub(channel)
  options = encode_params({
  })

  req = pb.CreateSessionRequest()
  response = stub.CreateSession(req)
  check(response.status)

  session = response.session

  response = stub.CreateModel(pb.CreateModelRequest(
    modelName="mlp",
    session=session,
    params=options,
  ))
  check(response.status)

  response = stub.DeleteSession(pb.DeleteSessionRequest(
    session=session,
  ))
  check(response.status)

if __name__ == '__main__':
  run()

