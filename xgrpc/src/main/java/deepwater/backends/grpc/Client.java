package deepwater.backends.grpc;

import ai.h2o.deepwater.backends.grpc.CreateModelRequest;
import ai.h2o.deepwater.backends.grpc.CreateModelResponse;
import ai.h2o.deepwater.backends.grpc.DeepWaterTrainBackendGrpc;
import ai.h2o.deepwater.backends.grpc.DeleteModelRequest;
import ai.h2o.deepwater.backends.grpc.LoadModelRequest;
import ai.h2o.deepwater.backends.grpc.LoadWeightsRequest;
import ai.h2o.deepwater.backends.grpc.ParamValue;
import ai.h2o.deepwater.backends.grpc.PredictRequest;
import ai.h2o.deepwater.backends.grpc.PredictResponse;
import ai.h2o.deepwater.backends.grpc.SaveModelRequest;
import ai.h2o.deepwater.backends.grpc.SaveWeightsRequest;
import ai.h2o.deepwater.backends.grpc.SetParametersRequest;
import ai.h2o.deepwater.backends.grpc.SetParametersResponse;
import ai.h2o.deepwater.backends.grpc.Status;
import ai.h2o.deepwater.backends.grpc.Tensor;
import ai.h2o.deepwater.backends.grpc.TrainRequest;
import ai.h2o.deepwater.backends.grpc.TrainResponse;
import com.google.common.primitives.Floats;
import com.google.protobuf.ByteString;
import deepwater.backends.BackendModel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class Client {
    private static final Logger logger = Logger.getLogger(Client.class.getName());

    private final DeepWaterTrainBackendGrpc.DeepWaterTrainBackendBlockingStub blockingStub;
    private final ManagedChannel channel;

    public Client(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true));
    }

    Client(ManagedChannelBuilder<?> channelBuilder) {
        channel = channelBuilder.build();
        blockingStub = DeepWaterTrainBackendGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public ParamValue asParam(Object value) throws Exception {
        if (value.getClass() == java.lang.String.class){
            return asParam((String) value);
        }
        else if (value.getClass() == java.lang.Long.class){
            return asParam((Long) value);
        }
        else if (value.getClass() == java.lang.Integer.class){
            return asParam((Integer) value);
        }
        else if (value.getClass() == java.lang.Boolean.class) {
            return asParam((Boolean) value);
        } else {
            throw new Exception("unsupported class value: "+value.getClass());
        }
    }

    public ParamValue asParam(Boolean value){
        return ParamValue.newBuilder().setB(value).build();
    }

    public ParamValue asParam(Integer value){
        return ParamValue.newBuilder().setI(value).build();
    }

    public ParamValue asParam(Long value){
        return ParamValue.newBuilder().setI(value).build();
    }

    public ParamValue asParam(String key){
        return ParamValue.newBuilder().setS(ByteString.copyFromUtf8(key)).build();
    }

    public Map<String, ParamValue> asParams(Map<String, Object> params) throws Exception {
        HashMap<String, ParamValue> map = new HashMap<>();
        for (String key: params.keySet()) {
            map.put(key, asParam(params.get(key)));
        }
        return map;
    }

    private void checkStatus(Status status) throws Exception {
        assert(status != null);
        if (!status.getOk()){
            throw new Exception(status.getMessage());
        }
    }

    public BackendModel createModel(String networkName, Map<String, Object> hashMap) throws Exception{
        CreateModelRequest req = CreateModelRequest.newBuilder()
                .setModelName(networkName)
                .putAllParams(asParams(hashMap))
                .build();
        CreateModelResponse response;
        response = blockingStub.createModel(req);
        checkStatus(response.getStatus());
        return new XGRPCBackendSession(response.getNetwork());
    }

    public void deleteModel(BackendModel model) throws Exception{
        DeleteModelRequest req = DeleteModelRequest.newBuilder()
                .setModel(asSession(model))
                .build();
        checkStatus(blockingStub.deleteModel(req));
    }

    public void loadModel(BackendModel model, String path) throws Exception{
        LoadModelRequest req = LoadModelRequest.newBuilder().build();
        Status status = blockingStub.loadModel(req);
        checkStatus(status);
    }


    public void saveModel(BackendModel model, String path) throws Exception{
        SaveModelRequest req = SaveModelRequest.newBuilder().build();

        Status status = blockingStub.saveModel(req);
        checkStatus(status);
    }


    public void saveWeights(BackendModel model, String path) throws Exception{
        SaveWeightsRequest req = SaveWeightsRequest.newBuilder().build();

        Status status = blockingStub.saveWeights(req);
        checkStatus(status);
    }


    public void loadWeights(BackendModel model, String path) throws Exception{
        LoadWeightsRequest req = LoadWeightsRequest.newBuilder().build();

        Status status = blockingStub.loadWeights(req);
        checkStatus(status);
    }


    private ai.h2o.deepwater.backends.grpc.BackendModel asSession(BackendModel model) {
        XGRPCBackendSession session = (XGRPCBackendSession) model;
        ai.h2o.deepwater.backends.grpc.BackendModel remoteModel = ai.h2o.deepwater.backends.grpc.BackendModel.newBuilder()
                .setState(session.getState())
                .build();
        return remoteModel;
    }

    private ai.h2o.deepwater.backends.grpc.BackendModel.Builder asSessionBuilder(BackendModel model) {
        XGRPCBackendSession session = (XGRPCBackendSession) model;
        return ai.h2o.deepwater.backends.grpc.BackendModel.newBuilder()
                .setState(session.getState());
    }


    public void setParameters(BackendModel model, Map<String, Object> params) throws Exception {

        SetParametersRequest req = SetParametersRequest.newBuilder()
                .setModel(asSessionBuilder(model))
                .putAllParams(asParams(params))
                .build();

        SetParametersResponse response = blockingStub.setParameters(req);

        checkStatus(response.getStatus());

    }


    public Map<String, float[]> train(BackendModel model, Map<String, float[]> m, String[] fetches) throws Exception {
        TrainRequest.Builder builder = TrainRequest.newBuilder();
        int i = 0;
        for (String key: m.keySet()) {
            Tensor.Builder tb = Tensor.newBuilder()
                    .addAllFloatValue(Floats.asList(m.get(key)));
            builder.setFeeds(i, tb);
            i++;
        }

        builder.setModel(asSessionBuilder(model));

        TrainResponse response = blockingStub.train(builder.build());

        checkStatus(response.getStatus());

        // unpack fetched tensors
        HashMap<String, float[]> results = new HashMap<>();
        for (Tensor t: response.getFetchesList()){
            results.put(t.getName(), Floats.toArray(t.getFloatValueList()));
        }

        return results;
    }


    public Map<String, float[]> predict(BackendModel model, Map<String, float[]> m, String[] fetches) throws Exception {
        PredictRequest.Builder builder= PredictRequest.newBuilder();

        builder.setModel(asSession(model));
        int i = 0;
        for (String key: m.keySet()) {
            Tensor.Builder tb = Tensor.newBuilder()
                    .addAllFloatValue(Floats.asList(m.get(key)));
                builder.setFeeds(i, tb);
            i++;
        }

        // add fetches
        i = 0;
        for(String key: fetches){
            Tensor.Builder tb = Tensor.newBuilder()
                    .setName(key);
            builder.setFetches(i, tb);
            i++;
        }

        PredictResponse response = blockingStub.predict(builder.build());

        checkStatus(response.getStatus());

        // unpack fetched tensors
        HashMap<String, float[]> results = new HashMap<>();
        for (Tensor t: response.getFetchesList()){
            results.put(t.getName(), Floats.toArray(t.getFloatValueList()));
        }

        return results;
    }

}
