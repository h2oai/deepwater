from __future__ import print_function

import grpc

import grpc_service_pb2 as pb


def check(resp):
  if resp.ok: return
  raise Exception(resp.message)


def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = pb.DeepWaterTrainBackendStub(channel)

  response = stub.CreateSession(pb.CreateSessionRequest({
    "GPUS" : 1,
    "CPUS" : 1,
  }))

  check(response)

  response = stub.CreateModel(pb.CreateModelRequest(
    name="mlp",
    session=response.session,
  ))
  check(response)


if __name__ == '__main__':
  run()

