package deepwater.backends.grpc;

import ai.h2o.deepwater.backends.grpc.*;
import com.google.common.primitives.Floats;
import com.google.protobuf.ByteString;
import deepwater.backends.BackendModel;
import deepwater.backends.RuntimeOptions;
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
        else if (value.getClass() == java.lang.Double.class){
            return asParam((Double) value);
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

    public ParamValue asParam(Double value){
        return ParamValue.newBuilder().setD(value).build();
    }

    public ParamValue asParam(Long value){
        return ParamValue.newBuilder().setI(value).build();
    }


    public ParamValue asParam(String key){
        return ParamValue.newBuilder().setS(key).build();
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

    public XGRPCBackendModel createModel(XGRPCBackendSession session,
                                         String networkName, Map<String, Object> hashMap) throws Exception{
        CreateModelRequest req = CreateModelRequest.newBuilder()
                .setSession(asSession(session))
                .setModelName(networkName)
                .putAllParams(asParams(hashMap))
                .build();
        CreateModelResponse response;
        response = blockingStub.createModel(req);
        checkStatus(response.getStatus());
        ai.h2o.deepwater.backends.grpc.BackendModel model = response.getModel();
        return new XGRPCBackendModel(model.getId().toByteArray(), model.getState().toByteArray());
    }

    public void deleteSession(XGRPCBackendSession session, BackendModel model) throws Exception{
        DeleteSessionRequest req = DeleteSessionRequest.newBuilder()
                .setSession(asSession(session))
                .build();
        checkStatus(blockingStub.deleteSession(req).getStatus());
    }

    public XGRPCBackendModel loadModel(XGRPCBackendSession session, String path) throws Exception{
        Map<String, Object> params = new HashMap<>();
        params.put("path", path);
        LoadModelRequest req = LoadModelRequest.newBuilder()
                .setSession(asSession(session))
                .putAllParams(asParams(params))
                .build();
        LoadModelResponse response = blockingStub.loadModel(req);
        checkStatus(response.getStatus());

        ai.h2o.deepwater.backends.grpc.BackendModel model = response.getModel();
        return new XGRPCBackendModel(model.getId().toByteArray(), model.getState().toByteArray());
    }

    private ByteString asByteString(String path) {
        return ByteString.copyFrom(path.getBytes());
    }

    private ByteString asByteString(byte[] data) {
        return ByteString.copyFrom(data);
    }


    public void saveModel(XGRPCBackendSession session, XGRPCBackendModel model, String path) throws Exception{
        Map<String, Object> params = new HashMap<>();
        params.put("path", path);
        SaveModelRequest req = SaveModelRequest.newBuilder()
                .setSession(asSession(session))
                .setModel(asModel(model))
                .putAllParams(asParams(params))
                .build();

        SaveModelResponse response = blockingStub.saveModel(req);
        checkStatus(response.getStatus());
    }


    public void saveModelVariables(XGRPCBackendSession session, XGRPCBackendModel model, String path) throws Exception{
        Map<String, Object> params = new HashMap<>();
        params.put("path", path);
        SaveModelVariablesRequest req = SaveModelVariablesRequest.newBuilder()
                .setModel(asModel(model))
                .setSession(asSession(session))
                .putAllParams(asParams(params))
                .build();

        SaveModelVariablesResponse response = blockingStub.saveModelVariables(req);
        checkStatus(response.getStatus());
    }


    public void loadModelVariables(XGRPCBackendSession session, XGRPCBackendModel model, String path) throws Exception{
        Map<String, Object> params = new HashMap<>();
        params.put("path", path);


        LoadModelVariablesRequest req = LoadModelVariablesRequest.newBuilder()
                .setModel(asModel(model))
                .setSession(asSession(session))
                .putAllParams(asParams(params))
                .build();
        LoadModelVariablesResponse response = blockingStub.loadModelVariables(req);
        checkStatus(response.getStatus());
    }


    private ai.h2o.deepwater.backends.grpc.BackendModel.Builder asModel(XGRPCBackendModel model) {
        return ai.h2o.deepwater.backends.grpc.BackendModel.newBuilder()
                .setId(asByteString(model.getUUID()))
                .setState(asByteString(model.getState()));
    }

    private Session.Builder asSession(XGRPCBackendSession session) {
        return ai.h2o.deepwater.backends.grpc.Session.newBuilder()
                .setHandle(session.getHandle());
    }


    public void setParameters(XGRPCBackendSession session, XGRPCBackendModel model, Map<String, Object> params) throws Exception {

        SetModelParametersRequest req = SetModelParametersRequest.newBuilder()
                .setSession(asSession(session))
                .setModel(asModel(model))
                .putAllParams(asParams(params))
                .build();

        SetModelParametersResponse response = blockingStub.setModelParameters(req);

        checkStatus(response.getStatus());

    }


    public Map<String, float[]> train(XGRPCBackendSession session, XGRPCBackendModel model, Map<String, float[]> m, String[] fetches) throws Exception {
        ExecuteRequest.Builder builder = ExecuteRequest.newBuilder();
        int i = 0;
        for (String key: m.keySet()) {
            Tensor.Builder tb = Tensor.newBuilder()
                    .setName(key)
                    .addAllFloatValue(Floats.asList(m.get(key)));
            builder.addFeeds(i, tb);
            i++;
        }

        // add fetches
        i = 0;
        for(String key: fetches){
            Tensor.Builder tb = Tensor.newBuilder()
                    .setName(key);
            builder.addFetches(i, tb);
            i++;
        }

        builder.setSession(asSession(session));
        builder.setModel(asModel(model));

        ExecuteResponse response = blockingStub.execute(builder.build());

        checkStatus(response.getStatus());

        // unpack fetched tensors
        HashMap<String, float[]> results = new HashMap<>();
        for (Tensor t: response.getFetchesList()){
            results.put(t.getName(), Floats.toArray(t.getFloatValueList()));
        }

        return results;
    }


    public Map<String, float[]> predict(XGRPCBackendSession session,
                                        XGRPCBackendModel model, Map<String, float[]> m, String[] fetches) throws Exception {
        ExecuteRequest.Builder builder= ExecuteRequest.newBuilder();
        builder.setSession(asSession(session));
        builder.setModel(asModel(model));

        // add feed
        int i = 0;
        for (String key: m.keySet()) {
            Tensor.Builder tb = Tensor.newBuilder()
                    .setName(key)
                    .addAllFloatValue(Floats.asList(m.get(key)));
                builder.addFeeds(i, tb);
            i++;
        }

        // add fetches
        i = 0;
        for(String key: fetches){
            Tensor.Builder tb = Tensor.newBuilder()
                    .setName(key);
            builder.addFetches(i, tb);
            i++;
        }

        ExecuteResponse response = blockingStub.execute(builder.build());

        checkStatus(response.getStatus());

        // unpack fetched tensors
        HashMap<String, float[]> results = new HashMap<>();
        for (Tensor t: response.getFetchesList()){
            results.put(t.getName(), Floats.toArray(t.getFloatValueList()));
        }

        return results;
    }

    public XGRPCBackendSession createSession(RuntimeOptions runtimeOptions) throws Exception {
        // TODO ignore options
        CreateSessionRequest req = CreateSessionRequest.newBuilder()
                .build();
        CreateSessionResponse response = blockingStub.createSession(req);
        checkStatus(response.getStatus());
        Session session = response.getSession();
        return new XGRPCBackendSession(session.getHandle());
    }
}
