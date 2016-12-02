package deepwater.backends.tensorflow;

import ai.h2o.deepwater.NetworkRequest;
import ai.h2o.deepwater.NetworkResponse;
import ai.h2o.deepwater.ParamValue;
import ai.h2o.deepwater.ServiceGrpc;
import ai.h2o.deepwater.PingRequest;
import ai.h2o.deepwater.Status;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.tensorflow.framework.MetaGraphDef;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;


public class Client {
    private static final Logger logger = Logger.getLogger(Client.class.getName());

    private final ServiceGrpc.ServiceBlockingStub blockingStub;
    private final ManagedChannel channel;

    /** Construct client connecting to HelloWorld server at {@code host:port}. */
    public Client(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true));
    }

    Client(ManagedChannelBuilder<?> channelBuilder) {
        channel = channelBuilder.build();
        blockingStub = ServiceGrpc.newBlockingStub(channel);
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

    public MetaGraphDef BuildNetwork(String networkName, Map<String, Object> hashMap) throws Exception {
        NetworkRequest.Builder requestBuilder = NetworkRequest.newBuilder();
        requestBuilder.putParams("name", asParam(networkName));

        for (String key: hashMap.keySet()) {
            requestBuilder.putParams(key, asParam(hashMap.get(key)));
        }

        NetworkResponse response;
        response = blockingStub.buildNetwork(requestBuilder.build());
        logger.info("buildNetwork: " + response);
        return response.getNetwork();
    }


    public void Ping() {
        PingRequest request = PingRequest.newBuilder().build();
        Status response;
        response = blockingStub.ping(request);
        logger.info("Ping: " + response);
    }

}
