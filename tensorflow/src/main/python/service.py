from concurrent import futures
import time


import grpc

import service_pb2


class DeepWaterServer(service_pb2.ServiceServicer):

  """Provides methods that implement functionality of DeepWater server."""

  def Ping(self, ctx):
    print("got request:", args)
    return service_pb2.Status()

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
