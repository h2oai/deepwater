package ai.h2o.deepwater.backends.grpc;

import static io.grpc.stub.ClientCalls.asyncUnaryCall;
import static io.grpc.stub.ClientCalls.asyncServerStreamingCall;
import static io.grpc.stub.ClientCalls.asyncClientStreamingCall;
import static io.grpc.stub.ClientCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.blockingServerStreamingCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncServerStreamingCall;
import static io.grpc.stub.ServerCalls.asyncClientStreamingCall;
import static io.grpc.stub.ServerCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.0.2)",
    comments = "Source: grpc_service.proto")
public class DeepWaterPredictBackendGrpc {

  private DeepWaterPredictBackendGrpc() {}

  public static final String SERVICE_NAME = "deepwater.DeepWaterPredictBackend";

  // Static method descriptors that strictly reflect the proto.
  @io.grpc.ExperimentalApi("https://github.com/grpc/grpc-java/issues/1901")
  public static final io.grpc.MethodDescriptor<ai.h2o.deepwater.backends.grpc.PredictRequest,
      ai.h2o.deepwater.backends.grpc.Status> METHOD_PREDICT =
      io.grpc.MethodDescriptor.create(
          io.grpc.MethodDescriptor.MethodType.UNARY,
          generateFullMethodName(
              "deepwater.DeepWaterPredictBackend", "Predict"),
          io.grpc.protobuf.ProtoUtils.marshaller(ai.h2o.deepwater.backends.grpc.PredictRequest.getDefaultInstance()),
          io.grpc.protobuf.ProtoUtils.marshaller(ai.h2o.deepwater.backends.grpc.Status.getDefaultInstance()));

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static DeepWaterPredictBackendStub newStub(io.grpc.Channel channel) {
    return new DeepWaterPredictBackendStub(channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static DeepWaterPredictBackendBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    return new DeepWaterPredictBackendBlockingStub(channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary and streaming output calls on the service
   */
  public static DeepWaterPredictBackendFutureStub newFutureStub(
      io.grpc.Channel channel) {
    return new DeepWaterPredictBackendFutureStub(channel);
  }

  /**
   */
  public static abstract class DeepWaterPredictBackendImplBase implements io.grpc.BindableService {

    /**
     */
    public void predict(ai.h2o.deepwater.backends.grpc.PredictRequest request,
        io.grpc.stub.StreamObserver<ai.h2o.deepwater.backends.grpc.Status> responseObserver) {
      asyncUnimplementedUnaryCall(METHOD_PREDICT, responseObserver);
    }

    @java.lang.Override public io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            METHOD_PREDICT,
            asyncUnaryCall(
              new MethodHandlers<
                ai.h2o.deepwater.backends.grpc.PredictRequest,
                ai.h2o.deepwater.backends.grpc.Status>(
                  this, METHODID_PREDICT)))
          .build();
    }
  }

  /**
   */
  public static final class DeepWaterPredictBackendStub extends io.grpc.stub.AbstractStub<DeepWaterPredictBackendStub> {
    private DeepWaterPredictBackendStub(io.grpc.Channel channel) {
      super(channel);
    }

    private DeepWaterPredictBackendStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DeepWaterPredictBackendStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new DeepWaterPredictBackendStub(channel, callOptions);
    }

    /**
     */
    public void predict(ai.h2o.deepwater.backends.grpc.PredictRequest request,
        io.grpc.stub.StreamObserver<ai.h2o.deepwater.backends.grpc.Status> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(METHOD_PREDICT, getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class DeepWaterPredictBackendBlockingStub extends io.grpc.stub.AbstractStub<DeepWaterPredictBackendBlockingStub> {
    private DeepWaterPredictBackendBlockingStub(io.grpc.Channel channel) {
      super(channel);
    }

    private DeepWaterPredictBackendBlockingStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DeepWaterPredictBackendBlockingStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new DeepWaterPredictBackendBlockingStub(channel, callOptions);
    }

    /**
     */
    public ai.h2o.deepwater.backends.grpc.Status predict(ai.h2o.deepwater.backends.grpc.PredictRequest request) {
      return blockingUnaryCall(
          getChannel(), METHOD_PREDICT, getCallOptions(), request);
    }
  }

  /**
   */
  public static final class DeepWaterPredictBackendFutureStub extends io.grpc.stub.AbstractStub<DeepWaterPredictBackendFutureStub> {
    private DeepWaterPredictBackendFutureStub(io.grpc.Channel channel) {
      super(channel);
    }

    private DeepWaterPredictBackendFutureStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DeepWaterPredictBackendFutureStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new DeepWaterPredictBackendFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<ai.h2o.deepwater.backends.grpc.Status> predict(
        ai.h2o.deepwater.backends.grpc.PredictRequest request) {
      return futureUnaryCall(
          getChannel().newCall(METHOD_PREDICT, getCallOptions()), request);
    }
  }

  private static final int METHODID_PREDICT = 0;

  private static class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final DeepWaterPredictBackendImplBase serviceImpl;
    private final int methodId;

    public MethodHandlers(DeepWaterPredictBackendImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_PREDICT:
          serviceImpl.predict((ai.h2o.deepwater.backends.grpc.PredictRequest) request,
              (io.grpc.stub.StreamObserver<ai.h2o.deepwater.backends.grpc.Status>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    return new io.grpc.ServiceDescriptor(SERVICE_NAME,
        METHOD_PREDICT);
  }

}
